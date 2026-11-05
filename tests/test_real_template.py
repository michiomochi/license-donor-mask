"""実様式テンプレート `templates/real_back_v1` の内容を固定する。

実写真・切り出し画像・変種画像・GT の JSON はリポジトリに置かない（テンプレートの
`template.json` / `ruling.png` / `print.png` だけ）。ここで固定するのは、その3ファイルが
「どう測って、どこまで fail-closed か」を表していること。

実写真は同梱しないので、このファイルのテストは画像ファイルを一切読まない。
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import ROOT, run_fixture

import fixtures_gen as fx
from donor_mask import field, mask
from donor_mask.config import ConfigError
from donor_mask.geom import rect_area, rect_clip, rect_union_bbox
from donor_mask.template import MAX_RULE_BAND_PX, Template, _validate_f_rules, load_templates

DIR = ROOT / "templates" / "real_back_v1"
FIELD = [40.0, 300.0, 971.0, 597.0]
PROTECTED = [[56.0, 52.0, 955.0, 288.0]]   # 下端 288: 上罫線（y 289〜296）は欄の上端の罫線で保護領域ではない
F_RULES = {
    "h_lines": [{"y": [289.0, 297.0], "x": [59.0, 950.0]},
                {"y": [545.0, 558.0], "x": [470.0, 930.0]},
                {"y": [588.0, 595.0], "x": [470.0, 930.0]}],
    "v_lines": [{"x": [52.0, 69.0], "y": [544.0, 572.0]},
                {"x": [947.0, 951.0], "y": [62.0, 293.0]}],
}


@pytest.fixture(scope="module")
def real(cfg):
    tmpls, _ = load_templates(ROOT / "templates", cfg)
    return next(t for t in tmpls if t.id == "real_back_v1")


def test_metadata_is_the_measured_spec(real):
    m = real.meta
    assert real.kind == "positive" and real.label == "back"
    assert m["synthetic"] is False               # 実写真から作った（生成器の出力ではない）
    assert m["size"] == [1011, 638]
    assert [float(v) for v in m["field_rect"]] == FIELD
    assert m["protected_rects"] == PROTECTED
    assert m["ignore_rects"] == [FIELD]          # 手書き・スタンプが乗りうる欄＝宣言ブロック全体
    assert m["window_expand"] == 0.1


def test_anchor_is_digits123_and_cues_are_the_minimum_set(real):
    """実様式に印刷された□/○の印は無いが、印刷数字 1/2/3 が実在する（A = digits123）。
    §1.8a の最小構成 T＋F＋A（A は局所構造）。F は閉じた枠ではなく罫線の線分（f_rules）の一致で判定する。"""
    m = real.meta
    assert m["anchor_type"] == "digits123"
    assert m["required_cues"] == ["T", "F", "A"]


def test_digit_anchor_is_the_measured_one(real):
    a = real.meta["anchor"]
    assert a["patches"] == ["digit_1.png", "digit_2.png", "digit_3.png"] and a["margin_px"] == 5
    assert a["centers"] == [[77.5, 372.5], [79.0, 402.0], [79.0, 432.0]]
    assert a["pitch"] == 30.0 and "side" not in a
    assert [p.shape for p in real.digit_patches] == [(29, 17), (30, 22), (30, 22)]   # (h, w)


def test_protected_gap_is_12px_and_at_least_the_required_minimum(real, cfg):
    """保護領域の下端 288 と欄の上端 300 の隙間 12px ≥ protected_min_gap_px + protected_clear_px（9px）。"""
    assert real.meta["protected_rects"][0][3] == 288.0 and real.meta["field_rect"][1] == 300.0
    gap = real.meta["field_rect"][1] - real.meta["protected_rects"][0][3]
    assert gap == 12.0 >= cfg.mask.protected_min_gap_px + cfg.mask.protected_clear_px


def test_declared_as_provisional(real):
    """3枚は別カード（裏面の印刷様式は共通）なので「様式ごとに5枚以上」を満たさない。暫定であることが JSON に出ていること。"""
    m = real.meta
    assert m["n_samples"] == 3 and m["min_samples_required"] == 5
    assert m["min_samples_met"] is False
    assert len(m["align_correlations"]) == 3 and min(m["align_correlations"]) >= 0.50
    assert "暫定" in m["note"]


def test_expected_mask_area_is_derived_from_field_rect(real, cfg):
    """expected_mask_area は field_rect と [mask] 余白から導ける値（手で書き換えていない）。"""
    pads = mask.base_pads(tuple(FIELD), cfg)
    assert real.meta["expected_mask_area"] == pytest.approx(rect_area(mask.padded(tuple(FIELD), pads)), abs=0.01)


def test_only_template_assets_are_committed(real):
    """サンプル画像・切り出し・GT の JSON を入れていないこと。数字パッチ（digit_*.png）はテンプレート資産。"""
    assert sorted(p.name for p in DIR.iterdir()) == ["digit_1.png", "digit_2.png", "digit_3.png",
                                                       "print.png", "ruling.png", "template.json"]
    assert sorted(json.loads((DIR / "template.json").read_text(encoding="utf-8"))) == sorted(real.meta)


def test_f_rules_are_the_measured_five_segments(real):
    """F は閉じた矩形ではなく、実測した5線分（横3・縦2）の一致で判定する。"""
    fr = real.meta["f_rules"]
    assert {k: fr[k] for k in ("h_lines", "v_lines")} == F_RULES
    assert len(fr["h_lines"]) == 3 and len(fr["v_lines"]) == 2
    assert "要校正" in fr["note"]


def test_protected_rects_exclude_the_top_ruling_of_the_field(real):
    """保護領域の下端は上罫線（y 289〜）の手前。上罫線は欄の上端であり保護対象ではない。"""
    top_rule_y0 = F_RULES["h_lines"][0]["y"][0]
    assert all(r[3] < top_rule_y0 for r in real.meta["protected_rects"])
    assert "上罫線" in real.meta["note"]


def test_note_states_digits123_and_the_design_value_of_y0(real):
    n = real.meta["note"]
    assert "digits123" in n and "罫線込みの和集合ではない" in n and "n=3" in n


def test_no_ruling_frame_can_satisfy_f(real, cfg):
    """この様式に「欄を囲む閉じた枠罫線」は無い、という実測をテンプレート自身で固定する。

    実様式の S6 は閉じた矩形を要求しない。`f_rules` を持つので線分モード（detect_rule_segments）で判定し、
    空欄の罫線マップに期待線分が全部乗れば成立する。一方、閉じた枠を探す従来モード（detect_frame）では
    罫線マップから取れる最大の枠はカード外周で、field_rect との IoU は面積比（約 0.43）にしかならず、
    IoU(T,F) >= iou_tf_min を満たす枠は無い。「枠が実在するから通る」経路は開かない。

    f_rules モードで線分が足りないときの理由は F_FRAME_NOT_FOUND（metric=f_rule_support）
    （test_f_rules_mode_reason_is_f_frame_not_found）。
    """
    rect, met = field.detect_frame(real.ruling, real.rect("field_rect"), cfg)
    if rect is not None:
        iou = met["iou_T"]
        assert iou < cfg.field.iou_tf_min, f"F が T と一致してしまう（ok 経路が開く）: {rect} iou={iou}"
    seg_rect, seg_met = field.detect_rule_segments(real.ruling, real.meta["f_rules"], cfg)
    assert seg_met["mode"] == "segments" and len(seg_met["rules"]) == 5
    assert all(r["support"] >= cfg.field.f_rules_support_min for r in seg_met["rules"]), seg_met["rules"]


def synthetic_f_rules_template(f_rules, field_rect):
    meta = {"id": "syn_frules", "kind": "positive", "label": "back", "size": [1011, 638], "window_expand": 0.1,
            "field_rect": list(field_rect), "anchor_type": "none", "anchor": None, "required_cues": ["T", "F"],
            "f_rules": f_rules, "protected_rects": [], "ignore_rects": [], "expected_mask_area": 1.0}
    return Template("syn_frules", "positive", "back", meta, np.zeros((638, 1011), np.uint8), None, [])


def draw_rules(f_rules, skip=None):
    """期待線分の中心に 4px の黒線を引いた位置合わせ済み画像（合成。実画像は使わない）。"""
    img = np.full((638, 1011, 3), 235, np.uint8)
    for i, r in enumerate(f_rules["h_lines"]):
        if skip == ("h", i):
            continue
        y = int(round((r["y"][0] + r["y"][1]) / 2))
        cv2.line(img, (int(r["x"][0]), y), (int(r["x"][1]), y), (0, 0, 0), 4)
    for i, r in enumerate(f_rules["v_lines"]):
        if skip == ("v", i):
            continue
        x = int(round((r["x"][0] + r["x"][1]) / 2))
        cv2.line(img, (x, int(r["y"][0])), (x, int(r["y"][1])), (0, 0, 0), 4)
    return img


FIELD_SYN = (100.0, 300.0, 900.0, 600.0)
F_RULES_SYN = {   # 宣言の外接 (80, 62, 922, 610) は欄 T の外（y<300, y>600）へはみ出す
    "h_lines": [{"y": [288.0, 292.0], "x": [80.0, 920.0]},
                {"y": [450.0, 454.0], "x": [300.0, 700.0]},
                {"y": [606.0, 610.0], "x": [300.0, 700.0]}],
    "v_lines": [{"x": [98.0, 102.0], "y": [450.0, 520.0]},
                {"x": [918.0, 922.0], "y": [62.0, 280.0]}],
}


def test_f_rules_mode_clips_f_to_the_field_rect(cfg):
    """f_rules モードの F は「宣言外接」ではなく rect_clip(宣言外接, field_rect)。R（T∪F∪A）は欄を超えない。

    罫線は欄の外へも伸びる（上の区切り罫線・上段右の縦罫線）。宣言外接をそのまま F にすると R が上段へ
    広がって保護領域を含んでしまう。
    """
    t = synthetic_f_rules_template(F_RULES_SYN, FIELD_SYN)
    cues, met, reasons = field.locate_field(draw_rules(F_RULES_SYN), t, cfg)
    assert reasons == [], reasons
    declared = met["F"]["rect_declared"]
    assert declared == [80.0, 62.0, 922.0, 610.0]
    assert cues["F"] == rect_clip(tuple(declared), FIELD_SYN) == (100.0, 300.0, 900.0, 600.0)
    assert met["F"]["rect_used"] == list(cues["F"])
    assert cues["R"] == rect_union_bbox([FIELD_SYN, cues["F"]]) == FIELD_SYN   # R は field_rect を超えない


def test_f_rules_mode_requires_every_segment(cfg):
    """5線分の1本が欠けても F は成立しない（縦罫線1本でも弱めない）。"""
    t = synthetic_f_rules_template(F_RULES_SYN, FIELD_SYN)
    for skip in (("h", 0), ("h", 1), ("h", 2), ("v", 0), ("v", 1)):
        cues, met, reasons = field.locate_field(draw_rules(F_RULES_SYN, skip=skip), t, cfg)
        assert cues is None, skip
        assert [r["code"] for r in reasons] == ["F_FRAME_NOT_FOUND"], skip


def ruling_map(f_rules, shift=None, drop_half=()):
    """合成の罫線マップ（0/255、1px 幅の線）。shift={(kind,i): (dx,dy)} で線をずらし、drop_half の線は前半だけ引く。"""
    rul = np.zeros((638, 1011), np.uint8)
    shift = shift or {}
    for kind, key, bk, sk in (("h", "h_lines", "y", "x"), ("v", "v_lines", "x", "y")):
        for i, r in enumerate(f_rules[key]):
            dx, dy = shift.get((kind, i), (0, 0))
            c = int(round((r[bk][0] + r[bk][1]) / 2))
            a, b = int(r[sk][0]), int(r[sk][1])
            if (kind, i) in drop_half:
                b = (a + b) // 2
            if kind == "h":
                rul[c + dy, a + dx:b + dx] = 255
            else:
                rul[a + dy:b + dy, c + dx] = 255
    return rul


def support_of(met, kind, i):
    return next(r["support"] for r in met["rules"] if r["kind"] == kind and r["index"] == i)


def test_f_rules_position_shift_within_tol_passes_and_beyond_tol_fails(cfg):
    """f_rules_tol_px の意味＝位置ずれの許容。バンド±tol 内なら成立、外へずらすと支持率 0 で不成立。"""
    tol = int(cfg.field.f_rules_tol_px)
    rect, met = field.detect_rule_segments(ruling_map(F_RULES_SYN), F_RULES_SYN, cfg)
    assert rect is not None and all(r["support"] == 1.0 for r in met["rules"])
    # h0 のバンドは [288,292]、線の中心は 290。+(2+tol) ずらすと 292+tol＝許容の縁（内側）
    rect, met = field.detect_rule_segments(ruling_map(F_RULES_SYN, shift={("h", 0): (0, tol + 2)}), F_RULES_SYN, cfg)
    assert rect is not None and support_of(met, "h", 0) == 1.0
    # バンド±tol の外（+20px）→ 支持率 0 で F 不成立
    rect, met = field.detect_rule_segments(ruling_map(F_RULES_SYN, shift={("h", 0): (0, 20)}), F_RULES_SYN, cfg)
    assert rect is None and met["found"] is False
    assert support_of(met, "h", 0) == 0.0
    # 縦罫線を横へ +20px ずらしても同じ
    rect, met = field.detect_rule_segments(ruling_map(F_RULES_SYN, shift={("v", 1): (20, 0)}), F_RULES_SYN, cfg)
    assert rect is None and support_of(met, "v", 1) == 0.0


def test_f_rules_half_missing_segment_has_half_support_and_fails(cfg):
    """線の半分が無い（支持率≈0.5）は f_rules_support_min=0.85 未満で不成立。"""
    rect, met = field.detect_rule_segments(ruling_map(F_RULES_SYN, drop_half=[("h", 1)]), F_RULES_SYN, cfg)
    assert support_of(met, "h", 1) == pytest.approx(0.5, abs=0.02)
    assert support_of(met, "h", 1) < cfg.field.f_rules_support_min
    assert rect is None and met["found"] is False
    assert all(support_of(met, k, i) == 1.0 for k, i in (("h", 0), ("h", 2), ("v", 0), ("v", 1)))


def test_f_rules_mode_reason_is_f_frame_not_found(cfg):
    """f_rules モードで罫線が無い（白紙）と、理由は F_FRAME_NOT_FOUND・metric=f_rule_support。

    一方、線分が離れていて閉じた枠を成さない画像でも、全線分が乗れば F は成立する（閉じた矩形を要求しない）。
    """
    t = synthetic_f_rules_template(F_RULES_SYN, FIELD_SYN)
    blank = np.full((638, 1011, 3), 235, np.uint8)
    cues, met, reasons = field.locate_field(blank, t, cfg)
    assert cues is None
    assert [(r["code"], r["metric"]) for r in reasons] == [("F_FRAME_NOT_FOUND", "f_rule_support")]
    assert reasons[0]["threshold"] == cfg.field.f_rules_support_min
    cues, met, reasons = field.locate_field(draw_rules(F_RULES_SYN), t, cfg)
    assert reasons == [] and met["F"]["mode"] == "segments"


# ---------------------------------------------------------------- f_rules の読み込み検証（異常系）
def _rules(**kw):
    return {**F_RULES, **kw}


def test_validate_f_rules_accepts_the_measured_spec():
    _validate_f_rules("t", {"f_rules": F_RULES}, [1011, 638])
    _validate_f_rules("t", {}, [1011, 638])   # f_rules は任意キー


@pytest.mark.parametrize("bad, match", [
    (_rules(h_lines=F_RULES["h_lines"][:2]), "h_lines は 3 本以上"),
    (_rules(v_lines=F_RULES["v_lines"][:1]), "v_lines は 2 本以上"),
    (_rules(h_lines=F_RULES["h_lines"][:2] + [{"y": [600.0, 600.0], "x": [470.0, 930.0]}]), "lo < hi"),
    (_rules(h_lines=F_RULES["h_lines"][:2] + [{"y": [588.0, 595.0], "x": [930.0, 470.0]}]), "lo < hi"),
    (_rules(v_lines=F_RULES["v_lines"][:1] + [{"x": [950.0, 947.0], "y": [62.0, 293.0]}]), "lo < hi"),
    (_rules(h_lines=F_RULES["h_lines"][:2] + [{"y": [630.0, 645.0], "x": [470.0, 930.0]}]), "キャンバス外"),
    (_rules(h_lines=F_RULES["h_lines"][:2] + [{"y": [588.0, 595.0], "x": [470.0, 1100.0]}]), "キャンバス外"),
    (_rules(v_lines=F_RULES["v_lines"][:1] + [{"x": [1000.0, 1015.0], "y": [62.0, 293.0]}]), "キャンバス外"),
    (_rules(v_lines=F_RULES["v_lines"][:1] + [{"x": [947.0, 951.0], "y": [-5.0, 293.0]}]), "キャンバス外"),
    ("not a dict", "オブジェクト"),
])
def test_validate_f_rules_rejects_weakened_or_broken_rules(bad, match):
    """F を弱める（線が少ない）・値が壊れている（lo>=hi・キャンバス外）f_rules は読み込みで ConfigError。"""
    with pytest.raises(ConfigError, match=match):
        _validate_f_rules("t", {"f_rules": bad}, [1011, 638])


def test_synthetic_fixture_does_not_pick_the_real_template(cfg, tmpls, tmp_path):
    """合成フィクスチャの実行で実様式テンプレートが選ばれない（誤って ok 経路に混ざらない）。"""
    rec, _, _ = run_fixture(tmp_path, fx.make({"seed": 1}), cfg, tmpls)
    assert rec["status"] == "ok"
    assert rec["stages"]["S3"]["best"]["template"].startswith("synthetic_")


def test_validate_f_rules_rejects_too_wide_band():
    """バンドを広げれば F を弱められる（しきい値を緩めるのと同じ）ので幅に上限 MAX_RULE_BAND_PX を課す。"""
    assert MAX_RULE_BAND_PX == 20
    wide = _rules(h_lines=F_RULES["h_lines"][:2] + [{"y": [570.0, 570.0 + MAX_RULE_BAND_PX + 1], "x": [470.0, 930.0]}])
    with pytest.raises(ConfigError, match="バンド幅が広すぎる.*20"):
        _validate_f_rules("t", {"f_rules": wide}, [1011, 638])
    wide_v = _rules(v_lines=F_RULES["v_lines"][:1] + [{"x": [930.0, 951.0], "y": [62.0, 293.0]}])
    with pytest.raises(ConfigError, match="バンド幅が広すぎる"):
        _validate_f_rules("t", {"f_rules": wide_v}, [1011, 638])


def test_validate_f_rules_accepts_the_widest_real_band():
    """現行テンプレートの最大 v0 の 17px は通る（上限ちょうどの 20px も通る）。"""
    v0 = F_RULES["v_lines"][0]["x"]
    assert v0[1] - v0[0] == 17 <= MAX_RULE_BAND_PX
    _validate_f_rules("t", {"f_rules": F_RULES}, [1011, 638])
    edge = _rules(v_lines=F_RULES["v_lines"][:1] + [{"x": [930.0, 930.0 + MAX_RULE_BAND_PX], "y": [62.0, 293.0]}])
    _validate_f_rules("t", {"f_rules": edge}, [1011, 638])
