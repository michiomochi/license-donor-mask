#!/usr/bin/env python3
"""【合成テンプレートの作成】tests/fixtures_gen.py の空欄カードから templates/ を作る。

これは設計書の `template-build`（実サンプル5枚以上から作る）ではない。v1 は合成レイアウト用のテンプレートだけを同梱する。
実物の様式のテンプレートは未作成（校正が必要。README「既知の限界」参照）。

使い方: .venv/bin/python tools/build_synthetic_templates.py
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fixtures_gen as fx  # noqa: E402

from donor_mask import layout, mask  # noqa: E402
from donor_mask.config import load_config  # noqa: E402
from donor_mask.geom import rect_area  # noqa: E402


def render_blank(kind):
    big = fx.draw_card(kind, boxes=3, blank=True)
    return cv2.resize(big, (fx.NW, fx.NH), interpolation=cv2.INTER_AREA)


def write_template(root: Path, tid, kind, label, img, cfg, meta_extra=None):
    d = root / tid
    d.mkdir(parents=True, exist_ok=True)
    dark = layout.dark_map(img, cfg)
    rul = layout.ruling_from_dark(dark, cfg)
    for name, arr in (("ruling.png", rul), ("print.png", dark)):
        ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        (d / name).write_bytes(buf.tobytes())
    meta = {"id": tid, "kind": kind, "label": label, "synthetic": True, "size": [fx.NW, fx.NH]}
    meta.update(meta_extra or {})
    (d / "template.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(root=None):
    cfg = load_config()
    root = Path(root) if root else ROOT / "templates"
    R = fx.FIELD_RECT
    pads = mask.base_pads(R, cfg)
    exp_area = rect_area(mask.padded(R, pads))
    back = {"field_rect": list(R), "window_expand": 0.10, "anchor_type": "box3",
            "anchor": {"side": fx.BOX_SIDE, "pitch": fx.BOX_PITCH, "centers": [list(c) for c in fx.BOX_CENTERS]},
            "required_cues": ["T", "F", "A"], "protected_rects": [list(r) for r in fx.PROTECTED],
            "ignore_rects": [list(r) for r in fx.IGNORE], "expected_mask_area": round(exp_area, 2),
            "note": "SYNTHETIC_TEST レイアウト用。実物の様式ではない"}
    write_template(root, "synthetic_back_v1", "positive", "back", render_blank("back"), cfg, back)
    write_template(root, "synthetic_front_v1", "negative", "front", render_blank("front"), cfg,
                   {"note": "表面相当の合成レイアウト（一致したら O_FRONT_SIDE）"})
    write_template(root, "synthetic_other_v1", "negative", "other", render_blank("other"), cfg,
                   {"note": "別様式の合成レイアウト（一致したら O_UNSUPPORTED_CARD）"})
    print("wrote", root)


if __name__ == "__main__":
    main()
