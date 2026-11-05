"""V7（マスク外のカード全面のインク。設計v2 §3 V7 / P1-L4）のテスト。

合成データのみ。実在の免許証画像は使わない。V7 の目的は「ok なのに欄の外に書かれた意思表示が残る」
経路を塞ぐこと。落とすべきもの（欄外の長い手書き）と、落としてはいけないもの（印刷物・
ignore_rects 内・マスク内・短いかすれ）を両方固定する。
"""
import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import ok_files, reason_codes, run_fixture

import fixtures_gen as fx
from donor_mask import layout, verify
from donor_mask.config import ConfigError, load_config
from donor_mask.geom import rect_poly
from donor_mask.template import Template

W, H = fx.NW, fx.NH
CARD = fx.CARD_COLOR
PEN = fx.PEN
PRINT = fx.INK
MASK = fx.FIELD_RECT                     # (330, 340, 950, 610)
# マスク・保護領域・ignore_rects のどれにも入らない領域（欄の右上の外側。印刷物は円が y 68〜112 にあるだけ）
FREE = (955, 140, 1005, 260)

# 欄外の長い手書き（長辺 64px）と、短いかすれ（長辺 14px）
LONG_STROKE = [(980, 170), (980, 230)]
SHORT_STROKE = [(962, 180), (972, 180)]


# ---------------------------------------------------------------- 単体（幾何と画像を直接作る）
def _template(print_map=None, ignore=None, protected=None) -> Template:
    meta = {"protected_rects": [list(r) for r in (fx.PROTECTED if protected is None else protected)]}
    ign = [tuple(r) for r in (fx.IGNORE if ignore is None else ignore)]
    return Template("v7_unit", "positive", "back", meta, np.zeros((H, W), np.uint8), print_map, ign)


def _geom(mask_rect=MASK) -> dict:
    """P_t2o = 恒等（作った画像がそのまま位置合わせ済み座標）。"""
    return {"P_t2o": np.eye(3).tolist(), "mask_rect": list(mask_rect),
            "mask_polygon": rect_poly(mask_rect).tolist()}


def _canvas(strokes=(), printed=(), color=CARD) -> np.ndarray:
    img = np.full((H, W, 3), color, np.uint8)
    for r in printed:  # 印刷物（罫線・文字もどき）
        cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), PRINT, 3)
    for pts in strokes:  # 手書き風（印刷物ではない＝print_map に入らない）
        cv2.polylines(img, [np.array(pts, np.int32)], False, PEN, 4, cv2.LINE_AA)
    return img


def _run_v7(cfg, img, tmpl, geom=None):
    return verify.v7_ink_outside(img, geom or _geom(), tmpl, cfg)


def test_v7_is_in_the_s8_verdicts_and_logged(cfg, tmpls, tmp_path):
    """V7 が S8 の判定に入り、ログの stages.S8 に残る（黙って足されていない状態にしない）。"""
    assert "V7" in verify.VERDICTS and verify.VERDICTS[-1] == "V7"
    f = fx.make({"angle": 4, "seed": 21})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert rec["status"] == "ok", rec["reasons"]
    assert "V7" in rec["stages"]["S8"] and rec["stages"]["S8"]["V7"]["pass"] is True


def test_printed_ink_is_subtracted_by_print_map(cfg):
    """印刷文字（print_map に含まれる暗画素）は V7 で拾わない。"""
    printed = [(FREE[0], FREE[1], FREE[2], FREE[3])]        # 欄外に印刷された枠（実在の印刷物のつもり）
    img = _canvas(printed=printed)
    print_map = layout.dark_map(img, cfg)                    # 空欄カードの暗画素＝print_map
    tmpl = _template(print_map)
    r = _run_v7(cfg, img, tmpl)
    assert r["pass"], r["metrics"]
    # 対照: 同じ場所に手書きを足すと拾う（このテストが「何も見ていないだけ」でないこと）
    img2 = _canvas(strokes=[LONG_STROKE], printed=printed)
    r2 = _run_v7(cfg, img2, _template(print_map))
    assert not r2["pass"] and r2["code"] == "V_INK_OUTSIDE"


def test_stroke_inside_ignore_rects_is_not_counted(cfg):
    """ignore_rects（記載事項欄）の中のストロークは拾わない。"""
    inside = [(500, 120), (500, 250)]                        # ignore_rects = (330,40)-(950,300) の中
    img = _canvas(strokes=[inside])
    r = _run_v7(cfg, img, _template())
    assert r["pass"], r["metrics"]
    # 対照: 同じ長さのストロークを ignore_rects の外（x>950）へ移すと拾う
    outside = [(FREE[0] + 25, FREE[1] + 20), (FREE[0] + 25, FREE[3] - 10)]
    r2 = _run_v7(cfg, _canvas(strokes=[outside]), _template())
    assert not r2["pass"] and r2["code"] == "V_INK_OUTSIDE"


def test_stroke_inside_the_mask_is_not_counted(cfg):
    """マスク内（塗る所）のインクは拾わない。"""
    inside = [(500, 400), (500, 500), (640, 500)]            # MASK = (330,340)-(950,610) の中
    r = _run_v7(cfg, _canvas(strokes=[inside]), _template())
    assert r["pass"], r["metrics"]


def test_ink_touching_the_card_edge_or_the_corners_is_not_counted(cfg):
    """カード縁 6px 以内・四隅の丸い角に入る成分は数えない（背景との境目の縞を手書きと数えない）。"""
    edge = [(3, 300), (3, 400)]                               # 縁 6px 以内（ink_map が 0 にする）
    corner = [(10, 10), (30, 40)]                             # 四隅 48px 四方
    r = _run_v7(cfg, _canvas(strokes=[edge, corner]), _template())
    assert r["pass"], r["metrics"]


# ---------------------------------------------------------------- 通し（パイプライン）
def test_long_handwriting_far_outside_the_mask_goes_to_review(cfg, tmpls, tmp_path):
    """1. マスク外に長辺 24px 以上の手書き → V_INK_OUTSIDE で review、ok/ は空。"""
    f = fx.make({"angle": 4, "seed": 21, "outside_stroke": [LONG_STROKE]})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert rec["status"] == "review", rec["reasons"]
    assert ok_files(od) == []
    assert rec["outputs"]["masked"] is None
    assert "V_INK_OUTSIDE" in reason_codes(rec)
    v7 = rec["stages"]["S8"]["V7"]
    assert v7["pass"] is False and v7["metrics"]["n_components"] >= 1
    box = v7["metrics"]["worst"][0]["bbox"]
    assert box[0] >= 950 and box[2] <= 1006 and box[1] >= 140 and box[3] <= 260, box


def test_short_stroke_outside_the_mask_still_passes(cfg, tmpls, tmp_path):
    """2. 同じ場所の短いかすれ（長辺 10px 程度）は拾わない → ok のまま。"""
    f = fx.make({"angle": 4, "seed": 21, "outside_stroke": [SHORT_STROKE]})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert rec["status"] == "ok", rec["reasons"]
    assert rec["stages"]["S8"]["V7"]["pass"] is True


def test_normal_positive_case_does_not_trip_v7(cfg, tmpls, tmp_path):
    """6. 正常な陽性ケース（既存フィクスチャ）が V7 で落ちない。"""
    f = fx.make({"angle": 4, "seed": 21})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert rec["status"] == "ok", rec["reasons"]
    v7 = rec["stages"]["S8"]["V7"]
    assert v7["pass"] is True and v7["metrics"]["n_components"] == 0
    assert ok_files(od) != []


def test_v7_cannot_be_disabled_by_config(tmp_path):
    """しきい値を緩めて V7 を無効化できない（0/1/極端な値は起動時に ConfigError）。"""
    src = (Path(__file__).resolve().parent.parent / "config.toml").read_text(encoding="utf-8")
    for k, bad in (("ink_min_len_px", "4"), ("ink_min_len_px", "400"),
                   ("ink_edge_margin_px", "40"), ("ink_corner_ignore_px", "-1")):
        lines = [f"{k} = {bad}" if ln.strip().startswith(k + " ") else ln for ln in src.splitlines()]
        p = tmp_path / f"bad_{k}_{bad}.toml"
        p.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)
