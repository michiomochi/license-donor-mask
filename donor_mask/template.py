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


CUES = ("T", "F", "A")
ANCHOR_TYPES = ("box3", "circle3", "none")


def _validate_cues_and_anchor(name: str, meta: dict):
    """required_cues と anchor_type の整合。

    - `required_cues` に F も A も入っていないテンプレート（= T だけ）は拒否する。T はテンプレートの
      位置そのものなので、それだけで塗ると「位置合わせが合っているか」を独立に確かめない経路になる。
      実様式でアンカーが合わないときの逃げ道にしない。
    - `anchor_type = "none"` は「印刷された選択肢マークが実在しない様式」を表す。このとき A は必須にできない。
    """
    cues = meta["required_cues"]
    if not isinstance(cues, list) or not cues or any(c not in CUES for c in cues):
        raise ConfigError(f"テンプレート {name}: required_cues は {list(CUES)} の空でない部分集合であること: {cues!r}")
    if not ({"F", "A"} & set(cues)):
        raise ConfigError(f"テンプレート {name}: required_cues に F か A を含めること"
                          f"（T だけのテンプレートはテンプレート位置だけで塗る経路になるため不可）: {cues!r}")
    at = meta["anchor_type"]
    if at not in ANCHOR_TYPES:
        raise ConfigError(f"テンプレート {name}: anchor_type は {list(ANCHOR_TYPES)} のいずれか: {at!r}")
    if at == "none":
        if "A" in cues:
            raise ConfigError(f"テンプレート {name}: anchor_type=\"none\" のとき required_cues に A は入れられません")
        return
    a = meta["anchor"]
    if not isinstance(a, dict) or any(k not in a for k in ("side", "pitch", "centers")):
        raise ConfigError(f"テンプレート {name}: anchor には side / pitch / centers が必要")
    if not isinstance(a["centers"], list) or len(a["centers"]) != 3:
        raise ConfigError(f"テンプレート {name}: anchor.centers は3点")
    if float(a["side"]) <= 0 or float(a["pitch"]) <= 0:
        raise ConfigError(f"テンプレート {name}: anchor.side / anchor.pitch は正")


def _num_pair(v) -> bool:
    return (isinstance(v, list) and len(v) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))


def _validate_f_rules(name: str, meta: dict, size, ):
    """f_rules の検証（任意キー）。F を「線分の一致」で判定する様式の期待位置。

    - h_lines（横罫線）3本以上・v_lines（縦罫線）2本以上を要求する。F を1本の線に弱める
      書き換えを読み込みで止めるため（設計 §1.8a の「手がかりを弱めない」）
    - 各線は {"y": [lo, hi], "x": [x0, x1]}（縦線は {"x": [lo, hi], "y": [y0, y1]}）。
      lo < hi、キャンバス内であること
    """
    fr = meta.get("f_rules")
    if fr is None:
        return
    if not isinstance(fr, dict):
        raise ConfigError(f"テンプレート {name}: f_rules はオブジェクトであること")
    W, H = float(size[0]), float(size[1])
    for key, band_key, span_key, minimum, limit in (("h_lines", "y", "x", 3, H), ("v_lines", "x", "y", 2, W)):
        lines = fr.get(key)
        if not isinstance(lines, list) or len(lines) < minimum:
            raise ConfigError(f"テンプレート {name}: f_rules.{key} は {minimum} 本以上（F を弱めない）")
        for i, rule in enumerate(lines):
            if not isinstance(rule, dict) or band_key not in rule or span_key not in rule:
                raise ConfigError(f"テンプレート {name}: f_rules.{key}[{i}] には {band_key} と {span_key} が必要")
            band, span = rule[band_key], rule[span_key]
            if not (_num_pair(band) and _num_pair(span)):
                raise ConfigError(f"テンプレート {name}: f_rules.{key}[{i}] の {band_key}/{span_key} は数値2つ")
            if band[1] <= band[0]:
                raise ConfigError(f"テンプレート {name}: f_rules.{key}[{i}].{band_key} は lo < hi")
            if span[1] <= span[0]:
                raise ConfigError(f"テンプレート {name}: f_rules.{key}[{i}].{span_key} は lo < hi")
            if not (0 <= band[0] <= band[1] <= limit):
                raise ConfigError(f"テンプレート {name}: f_rules.{key}[{i}].{band_key} がキャンバス外")
            other = W if key == "h_lines" else H
            if not (0 <= span[0] <= span[1] <= other):
                raise ConfigError(f"テンプレート {name}: f_rules.{key}[{i}].{span_key} がキャンバス外")


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
            _validate_cues_and_anchor(d.name, meta)
            _validate_f_rules(d.name, meta, meta.get("size") or [W, H])
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
