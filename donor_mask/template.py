"""様式テンプレート（JSON + 罫線マップPNG + 印刷物マップPNG）の読み込み。

templates/<id>/template.json   座標は全て正規化キャンバス(1011x638)の画素座標
templates/<id>/ruling.png      空欄カードの罫線マップ（S2 署名・S4 ECC 用）
templates/<id>/print.png       空欄カードの暗画素マップ（境界インクで印刷物を除くため）
kind: "positive"（対象の裏面様式）/ "negative"（表面など、一致したら review にする対象外様式）
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import layout
from .config import ConfigError


@dataclass
class Template:
    id: str
    kind: str
    label: str
    meta: dict
    ruling: np.ndarray
    print_map: np.ndarray | None
    ignore_rects: list = field(default_factory=list)
    sig: np.ndarray | None = None

    @property
    def positive(self) -> bool:
        return self.kind == "positive"

    def rect(self, key):
        return tuple(float(v) for v in self.meta[key])


def _read_png(p: Path):
    im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ConfigError(f"テンプレート画像を読めません: {p}")
    return im


def load_templates(root, cfg):
    """(templates(sorted by id), templates_sha256)。positive が1つも無ければ ConfigError。"""
    root = Path(root)
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "template.json").exists()) if root.is_dir() else []
    if not dirs:
        raise ConfigError(f"テンプレートがありません: {root}")
    h = hashlib.sha256()
    out = []
    for d in dirs:
        try:
            meta = json.loads((d / "template.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigError(f"テンプレートJSON不正: {d}: {e}") from e
        for fn in sorted(p.name for p in d.iterdir() if p.is_file()):
            h.update(f"{d.name}/{fn}".encode())
            h.update((d / fn).read_bytes())
        for k in ("id", "kind", "label"):
            if k not in meta:
                raise ConfigError(f"テンプレート {d.name}: キー {k} がありません")
        if meta["kind"] not in ("positive", "negative"):
            raise ConfigError(f"テンプレート {d.name}: kind は positive/negative")
        ruling = _read_png(d / "ruling.png")
        W, H = int(cfg.canvas.width), int(cfg.canvas.height)
        if ruling.shape != (H, W):
            raise ConfigError(f"テンプレート {d.name}: ruling.png は {W}x{H} であること")
        pm = _read_png(d / "print.png") if (d / "print.png").exists() else None
        if meta["kind"] == "positive":
            for k in ("field_rect", "window_expand", "anchor_type", "anchor", "required_cues",
                      "protected_rects", "ignore_rects", "expected_mask_area"):
                if k not in meta:
                    raise ConfigError(f"テンプレート {d.name}: キー {k} がありません")
            if pm is None:
                raise ConfigError(f"テンプレート {d.name}: print.png がありません")
        t = Template(meta["id"], meta["kind"], meta["label"], meta, ruling, pm,
                     [tuple(r) for r in meta.get("ignore_rects", [])])
        t.sig = layout.signature(ruling, t.ignore_rects, cfg)
        out.append(t)
    if not any(t.positive for t in out):
        raise ConfigError("positive テンプレートが1つもありません")
    return out, h.hexdigest()


def window_rect(t: Template, cfg) -> tuple:
    """探索窓: field_rect をカード幅・高さの window_expand ずつ外へ広げ、キャンバスにクランプ。"""
    W, H = float(cfg.canvas.width), float(cfg.canvas.height)
    x0, y0, x1, y1 = t.rect("field_rect")
    e = float(t.meta["window_expand"])
    return (max(0.0, x0 - e * W), max(0.0, y0 - e * H), min(W, x1 + e * W), min(H, y1 + e * H))
