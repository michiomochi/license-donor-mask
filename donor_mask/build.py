"""設計書 5.4 C1: 実サンプルから様式テンプレートを作る（`template-build`）。

入力は `rectify` で 1011x638 に正規化済みの**空欄**カード画像（手書き・スタンプ無し）。
実写真そのものは入力にしない。書き出すのは `template.json` / `ruling.png` / `print.png` だけで、
サンプル画像・切り出し画像はいっさい複製しない（テンプレートは罫線と座標だけを持つ）。

手順:
  1. 1枚目を基準に、2枚目以降を**罫線マップ同士の ECC(AFFINE)** で位置合わせする。
     相関が `cfg.calibrate.align_max_correlation` 未満なら「別様式が混ざっている」としてエラー
     （違う様式を平均して潰さない。c3 の下半分 1.6% ずれのような残差は平均で吸収してよい）。
  2. 罫線マップ = 位置合わせ後の ruling の平均を `avg_threshold` で二値化。印刷物マップ（print.png）
     も同じく暗画素の平均を二値化する（S7 の境界インクで印刷物を差し引くのに使う）。
  3. 座標（field_rect / anchor / protected_rects / ignore_rects）は**測定値**なので呼び出し側が必ず与える。
     このモジュールは座標を推測しない（推測で埋めると「塗ったが違う場所」を通す経路になる）。
  4. サンプル数が `min_samples` 未満なら **エラー**（設計書の「様式ごとに5枚以上」を緩めない）。
     `allow_fewer=True`（CLI `--allow-fewer`）を明示したときだけ `min_samples_met: false` を書いて
     暫定テンプレートとして通す。暫定でも他の検証（ローダー・必須キー・required_cues）は同じ。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import layout
from .config import ConfigError
from .geom import rect_area
from .mask import base_pads, padded
from .template import _validate_cues_and_anchor, load_templates

META_KEYS = ("field_rect", "anchor_type", "anchor", "required_cues", "protected_rects", "ignore_rects",
             "window_expand", "expected_mask_area", "note", "f_rules")


def _read_gray(p: Path) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ConfigError(f"サンプル画像を読めません: {p}")
    return im


def align_samples(images, cfg, ignore_rects):
    """1枚目を基準に ECC(AFFINE) で位置合わせした画像列と、各サンプルの相関を返す。

    ECC の warp 行列は「基準(=template)座標 -> そのサンプル座標」なので、そのまま
    `geom.warp_to_frame(sample, warp, W, H)` に渡すと基準座標系の画像が得られる。
    """
    e = cfg.ecc
    W, H = int(cfg.canvas.width), int(cfg.canvas.height)
    sg = float(e.blur_sigma)
    ref_rul = layout.zero_rects(layout.ruling_map(images[0], cfg), ignore_rects)
    ref_f = cv2.GaussianBlur(ref_rul.astype(np.float32) / 255.0, (0, 0), sg)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, int(e.iterations), float(e.eps))
    out, ccs = [images[0]], [1.0]
    for i, im in enumerate(images[1:], start=1):
        cur_rul = layout.zero_rects(layout.ruling_map(im, cfg), ignore_rects)
        cur_f = cv2.GaussianBlur(cur_rul.astype(np.float32) / 255.0, (0, 0), sg)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(ref_f, cur_f, warp, cv2.MOTION_AFFINE, crit, None, int(e.gauss_filt_size))
        except cv2.error as ex:
            raise ConfigError(f"サンプル {i + 1} 枚目の位置合わせに失敗しました"
                              f"（別様式の可能性）: {str(ex).splitlines()[0][:120]}") from ex
        lim = float(cfg.calibrate.align_max_correlation)
        if float(cc) < lim:
            raise ConfigError(f"サンプル {i + 1} 枚目の位置合わせ相関が低すぎます: {float(cc):.3f} < {lim}"
                              "（別様式が混ざっています。様式ごとに分けてください）")
        out.append(cv2.warpAffine(im, warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
        ccs.append(float(cc))
    return out, ccs


def _average_binarize(maps, cfg) -> np.ndarray:
    """0/255 マップの平均を avg_threshold で二値化した uint8（0/255）。"""
    thr = float(cfg.calibrate.avg_threshold) * 255.0
    avg = np.mean([m.astype(np.float32) for m in maps], axis=0)
    return np.where(avg >= thr, 255, 0).astype(np.uint8)


def resolve_spec(spec: dict) -> dict:
    """測定値のうち、テンプレートに書くキーだけを取り出す（足りないキーは ConfigError）。"""
    missing = [k for k in META_KEYS if k in ("field_rect", "anchor_type", "required_cues",
                                             "protected_rects", "ignore_rects") and k not in spec]
    if missing:
        raise ConfigError(f"テンプレートの測定値が足りません: {missing}（field_rect などは推測せず必ず与える）")
    return {k: spec[k] for k in META_KEYS if k in spec}


def build_template(samples, out_dir, cfg, *, template_id, label="back", kind="positive", spec=None,
                   allow_fewer=False, min_samples=None) -> dict:
    """正規化済みサンプルからテンプレートを1つ作る。戻り値は書いた template.json の内容。"""
    out_dir = Path(out_dir)
    W, H = int(cfg.canvas.width), int(cfg.canvas.height)
    imgs = [_read_gray(Path(p)) for p in samples]
    if not imgs:
        raise ConfigError("サンプルがありません（正規化済みの空欄カード画像が必要）")
    for p, im in zip(samples, imgs):
        if im.shape[:2] != (H, W):
            raise ConfigError(f"サンプル {p} が {W}x{H} ではありません: {im.shape[1]}x{im.shape[0]}"
                              "（先に `rectify` で正規化してください）")
    meta = resolve_spec(dict(spec or {}))
    ms = int(cfg.calibrate.min_samples if min_samples is None else min_samples)
    met = len(imgs) >= ms
    if not met and not allow_fewer:
        raise ConfigError(f"サンプルが {len(imgs)} 枚で足りません（min_samples={ms}）。"
                          "設計書 5.4 C1 は様式ごとに5枚以上を要求します。"
                          "暫定テンプレートとして書くなら --allow-fewer を明示してください")

    aligned, ccs = align_samples(imgs, cfg, meta["ignore_rects"])
    ruling = _average_binarize([layout.ruling_map(im, cfg) for im in aligned], cfg)
    print_map = _average_binarize([layout.dark_map(im, cfg) for im in aligned], cfg)

    exp = meta.get("expected_mask_area")
    if exp is None:
        R = [float(v) for v in meta["field_rect"]]
        exp = rect_area(padded(R, base_pads(R, cfg)))
    full = {"id": template_id, "kind": kind, "label": label, "synthetic": False,
            "size": [W, H], "window_expand": float(meta.get("window_expand", 0.10)),
            "field_rect": [float(v) for v in meta["field_rect"]],
            "anchor_type": meta["anchor_type"],
            "anchor": meta.get("anchor"),
            "required_cues": list(meta["required_cues"]),
            "f_rules": meta.get("f_rules"),
            "protected_rects": [[float(v) for v in r] for r in meta["protected_rects"]],
            "ignore_rects": [[float(v) for v in r] for r in meta["ignore_rects"]],
            "expected_mask_area": round(float(exp), 2),
            "n_samples": len(imgs), "min_samples_required": ms, "min_samples_met": bool(met),
            "align_correlations": [round(c, 4) for c in ccs],
            "note": meta.get("note", "")}
    _validate_cues_and_anchor(template_id, full)  # 書き出す前にローダーと同じ規則で落とす

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in (("ruling.png", ruling), ("print.png", print_map)):
        ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise ConfigError(f"テンプレート画像を書けません: {out_dir / name}")
        (out_dir / name).write_bytes(buf.tobytes())
    (out_dir / "template.json").write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 書いたものをローダーで読み直す（読めないテンプレートを残さない）
    tmpls, _ = load_templates(out_dir.parent, cfg)
    if template_id not in [t.id for t in tmpls]:
        raise ConfigError(f"生成したテンプレート {template_id} をローダーが読めません: {out_dir}")
    return full
