"""A(digits123): 印刷数字 1/2/3 のパッチ照合と、テンプレートのロード時検査（設計 §1.8a / §1.8b）。

数字パッチは「印刷様式部品の位置合わせ」であって文字認識ではない（OCR・学習済みモデルは使わない）。
画像は全て tests/fixtures_gen.py の合成データ。実写真・実カードの切り出しは使わない。
"""
import copy
import dataclasses
import importlib.util
import json
import shutil

import cv2
import numpy as np
import pytest
from conftest import ROOT, run_fixture

import fixtures_gen as fx
from donor_mask import field, verify
from donor_mask.config import ConfigError, load_config
from donor_mask.template import _validate_cues_and_anchor, _validate_protected_gap, load_templates, window_rect

REAL_DIR = ROOT / "templates" / "real_back_v1"
MARGIN = 5


# ---------------------------------------------------------------- 合成の数字パッチ
def blank_gray() -> np.ndarray:
    big = fx.draw_card("back", boxes=3, blank=True)
    return cv2.cvtColor(cv2.resize(big, (fx.NW, fx.NH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def digit_boxes(g):
    """フィクスチャが「N DUMMY OPTION TEXT」の先頭に描く印刷数字 N の外接矩形（両端含む）x3。"""
    out = []
    for y in fx.BOX_YS:
        base = y + 22
        reg = g[base - 18:base + 5, 398:411]
        ys, xs = np.nonzero(reg < 150)
        out.append((398 + int(xs.min()), base - 18 + int(ys.min()), 398 + int(xs.max()), base - 18 + int(ys.max())))
    return out


@pytest.fixture(scope="module")
def synth():
    g = blank_gray()
    boxes = digit_boxes(g)
    patches = [g[y0 - MARGIN:y1 + MARGIN + 1, x0 - MARGIN:x1 + MARGIN + 1].copy() for x0, y0, x1, y1 in boxes]
    centers = [[(x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0] for x0, y0, x1, y1 in boxes]
    anchor = {"patches": ["digit_1.png", "digit_2.png", "digit_3.png"], "margin_px": MARGIN,
              "pitch": centers[1][1] - centers[0][1], "centers": centers}
    digit_h = float(np.mean([p.shape[0] for p in patches])) - 2 * MARGIN
    return {"gray": g, "boxes": boxes, "patches": patches, "centers": centers, "anchor": anchor, "digit_h": digit_h}


def canvas_with(synth, order=(0, 1, 2), dx=0, extra_dx=None):
    """無地キャンバスにパッチを貼った画像。order は y の昇順に貼る数字（0=「1」）。extra_dx があれば3連をもう1組貼る。"""
    g = np.full((fx.NH, fx.NW), 235, np.uint8)
    for dxs in [dx] + ([extra_dx] if extra_dx is not None else []):
        for slot, which in enumerate(order):
            x0, y0, x1, y1 = synth["boxes"][slot]          # 位置はスロット（元の3連の位置）、中身が which
            p = synth["patches"][which]
            h, w = p.shape
            ty0 = y0 - MARGIN
            tx0 = x0 - MARGIN + dxs
            g[ty0:ty0 + h, tx0:tx0 + w] = p
    return g


def win_of(g):
    return (0.0, 0.0, float(g.shape[1]), float(g.shape[0]))


def digits_template(cfg, synth):
    """合成テンプレート（synthetic_back_v1）の anchor を digits123 に差し替えたもの。"""
    tmpls, _ = load_templates(ROOT / "templates", cfg)
    base = next(t for t in tmpls if t.id == "synthetic_back_v1")
    meta = copy.deepcopy(base.meta)
    meta["anchor_type"] = "digits123"
    meta["anchor"] = copy.deepcopy(synth["anchor"])
    meta["required_cues"] = ["T", "F", "A"]        # 設計 §1.8a の最小構成（digits123 では A が必須）
    return dataclasses.replace(base, meta=meta, digit_patches=[p.copy() for p in synth["patches"]]), tmpls


# ---------------------------------------------------------------- 検出の単体
def test_one_triple_on_the_synthetic_card(cfg, synth):
    g = synth["gray"]
    win = window_rect(digits_template(cfg, synth)[0], cfg)
    peaks = field.digit_peaks(g, synth["patches"], win, cfg)
    assert len(peaks) == 3 and all(len(p) >= 1 for p in peaks)
    assert all(set(d) >= {"cx", "cy", "rect", "value"} for p in peaks for d in p)
    triples = field.find_digit_triples(peaks, synth["anchor"], synth["digit_h"], cfg)
    assert len(triples) == 1
    tr = triples[0]
    assert [round(d["value"], 3) for d in tr] == [1.0, 1.0, 1.0]           # 切り出し元と同じ画像なので完全一致
    assert [[d["cx"], d["cy"]] for d in tr] == synth["centers"]            # パッチ中心＝期待中心


def strict(cfg):
    """合成の数字は 9x11px と小さく、別の数字同士でも TM_CCOEFF_NORMED が約 0.75 になる（字形の差が小さい）。
    並び順の検査は「別の数字は一致しない」ことが前提なので、このテストでだけ s_digit を 0.9 にする
    （既定値を緩める方向ではなく、厳しくする方向。実カードの数字の識別は別途実測）。"""
    return cfg.override("field", s_digit=0.9)


def test_synthetic_digits_are_similar_to_each_other_at_the_default_threshold(cfg, synth):
    """上の strict() の前提の固定: 既定の s_digit=0.6 だと合成の別の数字も極大に入る。"""
    g = synth["gray"]
    peaks = field.digit_peaks(g, [synth["patches"][0]], win_of(g), cfg)
    assert len(peaks[0]) == 3          # 自分の位置（1.0）に加えて「2」「3」の位置も ~0.75 で拾う
    peaks = field.digit_peaks(g, [synth["patches"][0]], win_of(g), strict(cfg))
    assert len(peaks[0]) == 1


def test_swapped_patches_or_image_order_give_zero_triples(cfg, synth):
    cfg = strict(cfg)
    g = synth["gray"]
    win = win_of(g)
    # パッチの順序を入れ替える（1,2,3 → 2,1,3）
    swapped = [synth["patches"][i] for i in (1, 0, 2)]
    peaks = field.digit_peaks(g, swapped, win, cfg)
    assert field.find_digit_triples(peaks, synth["anchor"], synth["digit_h"], cfg) == []
    # 画像側の並びを逆（3,2,1）にする
    rev = canvas_with(synth, order=(2, 1, 0))
    peaks = field.digit_peaks(rev, synth["patches"], win, cfg)
    assert field.find_digit_triples(peaks, synth["anchor"], synth["digit_h"], cfg) == []
    # 対照: 正しい並びなら1組
    ok = canvas_with(synth)
    peaks = field.digit_peaks(ok, synth["patches"], win, cfg)
    assert len(field.find_digit_triples(peaks, synth["anchor"], synth["digit_h"], cfg)) == 1


def test_reversed_order_is_rejected_by_the_order_margin_at_the_default_threshold(cfg, synth):
    """逆順（3,2,1）の並びは、既定の s_digit=0.6 でも3連にはなるが、順序の判定（digit_margin_min）で落ちる。

    s_digit だけでは数字どうしの照合値（この合成では約 0.75）が下限を超えるため、順序の判定は
    digit_order_margin が担う。ここでは「幾何では1組」「順序では不合格」を別々に固定する。
    """
    rev = canvas_with(synth, order=(2, 1, 0))
    peaks = field.digit_peaks(rev, synth["patches"], win_of(rev), cfg)
    triples = field.find_digit_triples(peaks, synth["anchor"], synth["digit_h"], cfg)
    assert len(triples) == 1                      # 既定のしきい値では幾何の条件は満たしてしまう
    _mat, worst = field.digit_order_margin(rev, synth["patches"], triples[0])
    assert worst < 0.0 and worst < cfg.field.digit_margin_min
    ok = canvas_with(synth)
    peaks = field.digit_peaks(ok, synth["patches"], win_of(ok), cfg)
    _mat, worst_ok = field.digit_order_margin(ok, synth["patches"], field.find_digit_triples(
        peaks, synth["anchor"], synth["digit_h"], cfg)[0])
    assert worst_ok > cfg.field.digit_margin_min


def test_duplicated_triple_gives_two_triples(cfg, synth):
    g = canvas_with(synth, extra_dx=400)
    peaks = field.digit_peaks(g, synth["patches"], win_of(g), cfg)
    triples = field.find_digit_triples(peaks, synth["anchor"], synth["digit_h"], cfg)
    assert len(triples) == 2


def test_pitch_out_of_tolerance_is_not_a_triple(cfg, synth):
    """期待ピッチが ±digit_pitch_tol を外れる並びは3連にしない（しきい値を緩めない）。"""
    g = canvas_with(synth)
    peaks = field.digit_peaks(g, synth["patches"], win_of(g), cfg)
    far = dict(synth["anchor"], pitch=synth["anchor"]["pitch"] * 1.5)
    assert field.find_digit_triples(peaks, far, synth["digit_h"], cfg) == []
    near = dict(synth["anchor"], pitch=synth["anchor"]["pitch"] * 1.10)   # tol 0.15 の内側
    assert len(field.find_digit_triples(peaks, near, synth["digit_h"], cfg)) == 1


def test_peak_count_is_capped_at_max_plus_one(cfg, synth):
    """極大が多すぎることを検出できるよう、1パッチあたり digit_max_peaks+1 個まで取る。"""
    g = np.full((fx.NH, fx.NW), 235, np.uint8)
    p = synth["patches"][0]
    h, w = p.shape
    for i in range(10):
        for j in range(5):
            g[10 + 40 * j:10 + 40 * j + h, 10 + 60 * i:10 + 60 * i + w] = p
    small = cfg.override("field", digit_max_peaks=7)
    peaks = field.digit_peaks(g, synth["patches"], win_of(g), small)
    assert len(peaks[0]) == 8
    assert len(field.digit_peaks(g, synth["patches"], win_of(g), cfg)[0]) == 50


def test_nms_suppresses_neighbours(cfg, synth):
    g = canvas_with(synth)
    peaks = field.digit_peaks(g, synth["patches"], win_of(g), cfg)
    r = int(cfg.field.digit_nms_px)
    for plist in peaks:
        for i, a in enumerate(plist):
            for b in plist[i + 1:]:
                assert max(abs(a["cx"] - b["cx"]), abs(a["cy"] - b["cy"])) > r


# ---------------------------------------------------------------- locate_field
def test_locate_field_returns_a_and_a_boxes(cfg, synth):
    t, _ = digits_template(cfg, synth)
    # 切り出し元と同じ合成カードなので、F（閉じた枠）は T と一致し、A は恒等変換になる
    img = cv2.resize(fx.draw_card("back", boxes=3, blank=True), (fx.NW, fx.NH), interpolation=cv2.INTER_AREA)
    cues, met, reasons = field.locate_field(img, t, cfg)
    assert reasons == [], reasons
    assert cues["A"] is not None and cues["A_boxes"] is not None and len(cues["A_boxes"]) == 3
    a = met["A"]
    assert a["anchor_type"] == "digits123" and a["s_digit"] == cfg.field.s_digit
    assert a["triples"] == 1 and len(a["peaks"]) == 3 and len(a["values"]) == 3 and len(a["positions"]) == 3
    assert a["scale"] == pytest.approx(1.0, abs=1e-6)
    assert cues["A"] == pytest.approx(t.rect("field_rect"), abs=1e-6)       # 恒等変換なので欄そのもの


@pytest.mark.parametrize("variant, code, metric", [
    ("none", "F_ANCHOR_NOT_FOUND", "digit_triples"),
    ("two", "F_ANCHOR_MULTIPLE", "digit_triples"),
])
def test_locate_field_digit_failures(cfg, synth, variant, code, metric):
    t, _ = digits_template(cfg, synth)
    base = cv2.resize(fx.draw_card("back", boxes=3, blank=True), (fx.NW, fx.NH), interpolation=cv2.INTER_AREA)
    if variant == "none":
        t = dataclasses.replace(t, digit_patches=[synth["patches"][i] for i in (1, 0, 2)])
        img = base
    else:
        img = base.copy()
        for slot in range(3):
            x0, y0, x1, y1 = synth["boxes"][slot]
            p = synth["patches"][slot]
            h, w = p.shape
            img[y0 - MARGIN:y0 - MARGIN + h, x0 - MARGIN + 400:x0 - MARGIN + 400 + w] = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
    cues, met, reasons = field.locate_field(img, t, strict(cfg))
    assert cues is None
    assert [(r["code"], r["metric"]) for r in reasons] == [(code, metric)]


def test_locate_field_too_many_peaks_is_multiple(cfg, synth):
    t, _ = digits_template(cfg, synth)
    base = cv2.resize(fx.draw_card("back", boxes=3, blank=True), (fx.NW, fx.NH), interpolation=cv2.INTER_AREA)
    img = base.copy()
    for slot in range(3):
        x0, y0, x1, y1 = synth["boxes"][slot]
        p = synth["patches"][slot]
        h, w = p.shape
        img[y0 - MARGIN:y0 - MARGIN + h, x0 - MARGIN + 400:x0 - MARGIN + 400 + w] = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
    tight = cfg.override("field", digit_max_peaks=1)      # 各パッチ2か所 → 上限超過
    cues, met, reasons = field.locate_field(img, t, tight)
    assert cues is None
    assert [(r["code"], r["metric"]) for r in reasons] == [("F_ANCHOR_MULTIPLE", "digit_peaks")]


def test_box3_path_is_unchanged(cfg):
    """合成テンプレート（box3）の S6 は従来どおり（digits123 を足しても経路を変えない）。"""
    tmpls, _ = load_templates(ROOT / "templates", cfg)
    t = next(t for t in tmpls if t.id == "synthetic_back_v1")
    assert t.meta["anchor_type"] == "box3" and t.digit_patches == []
    img = cv2.resize(fx.draw_card("back", boxes=3, blank=True), (fx.NW, fx.NH), interpolation=cv2.INTER_AREA)
    cues, met, reasons = field.locate_field(img, t, cfg)
    assert reasons == [] and met["A"]["triples"] == 1 and met["A"]["singles"] >= 3


# ---------------------------------------------------------------- パイプライン（S6 → S7 → V4）
def test_pipeline_ok_with_digits123_and_v4_checks_the_digit_boxes(cfg, synth, tmp_path):
    t, tmpls = digits_template(cfg, synth)
    templates = [t if x.id == t.id else x for x in tmpls]
    rec, out, _ = run_fixture(tmp_path, fx.make({"seed": 1, "jpeg": None}), cfg, (templates, "x"))
    assert rec["status"] == "ok", rec["reasons"]
    v4 = rec["stages"]["S8"]["V4"]["metrics"]
    assert v4["anchor"] == "digits123" and v4["singles"] == 0 and v4["triples"] == 0
    # 位置決めに使った数字の3矩形がマスクに入っていて、その中が塗られている（照合値が下限未満）
    assert v4["digit_boxes"] == 3 and v4["digit_boxes_in_mask"] is True and v4["digit_regions"] == 9
    assert v4["digit_max"] < cfg.verify.digit_absent_max


def test_v4_fails_when_digits_remain_inside_the_mask(cfg, synth, tmp_path):
    """塗られていない画像（数字が残っている）は、数字の3矩形の照合が下限を超えて V_REDETECT_TRACE。"""
    t, tmpls = digits_template(cfg, synth)
    templates = [t if x.id == t.id else x for x in tmpls]
    fixture = fx.make({"seed": 1, "jpeg": None})
    rec, out, _ = run_fixture(tmp_path, fixture, cfg, (templates, "x"))
    assert rec["status"] == "ok"
    geom = rec["geometry"]
    mask = verify.make_mask(fixture.img.shape, geom["mask_polygon"], int(cfg.mask.grow_px))
    res = verify.v4_redetect(fixture.img, geom, t, cfg, mask)     # 原本（塗っていない）を渡す
    assert res["metrics"]["digit_max"] >= cfg.verify.digit_absent_max
    assert res["pass"] is False and res["code"] == "V_REDETECT_TRACE"


def test_v4_fails_when_the_mask_misses_the_digit_boxes(cfg, synth, tmp_path):
    """マスク矩形が数字の3矩形を覆っていない（＝塗り残し）なら、塗りが一様でも V_REDETECT_TRACE。

    S7 が保護領域との隙間で余白を縮めたときにマスクが内側へ寄る経路があるので、到達しうる状態。
    """
    t, tmpls = digits_template(cfg, synth)
    templates = [t if t.id == t.id else x for x in tmpls]
    fixture = fx.make({"seed": 1, "jpeg": None})
    rec, out, _ = run_fixture(tmp_path, fixture, cfg, (templates, "x"))
    assert rec["status"] == "ok"
    geom = copy.deepcopy(rec["geometry"])
    x0, y0, x1, y1 = geom["mask_rect"]
    geom["mask_rect"] = [600.0, y0, x1, y1]        # 桁の左端がマスクの外へ出る
    mask = verify.make_mask(fixture.img.shape, geom["mask_polygon"], int(cfg.mask.grow_px))
    res = verify.v4_redetect(cv2.imdecode(np.fromfile(rec["outputs"]["masked"], np.uint8), cv2.IMREAD_COLOR),
                             geom, t, cfg, mask)
    assert res["metrics"]["digit_boxes_in_mask"] is False
    assert res["pass"] is False and res["code"] == "V_REDETECT_TRACE"


# ---------------------------------------------------------------- §1.8a: required_cues の最小構成
BOX3 = {"side": 28, "pitch": 60, "centers": [[1, 1], [1, 2], [1, 3]]}
DIGITS = {"patches": ["a.png", "b.png", "c.png"], "pitch": 30.0, "centers": [[1, 1], [1, 2], [1, 3]]}


@pytest.mark.parametrize("cues", [["T"], ["T", "F"], ["T", "A"]])
def test_minimum_cue_set_is_enforced(cues):
    for at, anchor in (("box3", BOX3), ("digits123", DIGITS)):
        with pytest.raises(ConfigError, match="§1.8a の最小構成") as e:
            _validate_cues_and_anchor("t", {"required_cues": cues, "anchor_type": at, "anchor": anchor})
        assert repr(cues) in str(e.value)


def test_cues_without_t_are_rejected():
    with pytest.raises(ConfigError, match="T"):
        _validate_cues_and_anchor("t", {"required_cues": ["F", "A"], "anchor_type": "box3", "anchor": BOX3})


def test_anchor_none_cannot_meet_the_minimum():
    for cues in (["T", "F", "A"], ["T", "F"], ["T"]):
        with pytest.raises(ConfigError, match="none"):
            _validate_cues_and_anchor("t", {"required_cues": cues, "anchor_type": "none", "anchor": None})


def test_full_cue_set_is_accepted_for_box3_and_digits123():
    _validate_cues_and_anchor("t", {"required_cues": ["T", "F", "A"], "anchor_type": "box3", "anchor": BOX3})
    _validate_cues_and_anchor("t", {"required_cues": ["T", "F", "A"], "anchor_type": "digits123", "anchor": DIGITS})
    _validate_cues_and_anchor("t", {"required_cues": ["T", "F", "A"], "anchor_type": "circle3", "anchor": BOX3})


@pytest.mark.parametrize("mut, match", [
    (lambda a: a.pop("patches"), "patches"),
    (lambda a: a.__setitem__("patches", ["a.png", "b.png"]), "patches"),
    (lambda a: a.__setitem__("patches", ["a.png", "b.png", 3]), "patches"),
    (lambda a: a.pop("centers"), "centers"),
    (lambda a: a.__setitem__("centers", [[1, 1], [1, 2]]), "centers"),
    (lambda a: a.__setitem__("centers", [[1, 1], [1, 2], [1]]), "centers"),
    (lambda a: a.pop("pitch"), "pitch"),
    (lambda a: a.__setitem__("pitch", 0), "pitch"),
    (lambda a: a.__setitem__("pitch", -30.0), "pitch"),
    (lambda a: a.__setitem__("margin_px", -1), "margin_px"),
])
def test_digits123_anchor_shape_is_validated(mut, match):
    a = copy.deepcopy(DIGITS)
    mut(a)
    with pytest.raises(ConfigError, match=match):
        _validate_cues_and_anchor("t", {"required_cues": ["T", "F", "A"], "anchor_type": "digits123", "anchor": a})


def test_digits123_does_not_require_side():
    assert "side" not in DIGITS
    _validate_cues_and_anchor("t", {"required_cues": ["T", "F", "A"], "anchor_type": "digits123", "anchor": DIGITS})


def copy_real(tmp_path, mutate=None, drop_files=()):
    """実テンプレートを tmp にコピーし、template.json を書き換える（ロード時検査の異常系用）。"""
    dst = tmp_path / "templates" / "real_back_v1"
    shutil.copytree(REAL_DIR, dst)
    if mutate:
        meta = json.loads((dst / "template.json").read_text(encoding="utf-8"))
        mutate(meta)
        (dst / "template.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    for f in drop_files:
        (dst / f).unlink()
    return dst.parent


def test_real_template_loads_with_digit_patches(cfg):
    tmpls, _ = load_templates(ROOT / "templates", cfg)
    real = next(t for t in tmpls if t.id == "real_back_v1")
    assert [p.shape for p in real.digit_patches] == [(29, 17), (30, 22), (30, 22)]
    assert all(p.dtype == np.uint8 and p.ndim == 2 for p in real.digit_patches)


@pytest.mark.parametrize("mut, drop, match", [
    (lambda m: m["anchor"].pop("patches"), (), "patches"),
    (None, ("digit_2.png",), "digit_2.png"),
    (lambda m: m["anchor"].__setitem__("centers", m["anchor"]["centers"][:2]), (), "centers"),
    (lambda m: m["anchor"].__setitem__("pitch", 0), (), "pitch"),
    (lambda m: m["anchor"].__setitem__("pitch", -1.0), (), "pitch"),
    (lambda m: m["anchor"].__setitem__("patches", ["digit_1.png", "digit_2.png", "nothing.png"]), (), "nothing.png"),
])
def test_digits123_template_load_errors(cfg, tmp_path, mut, drop, match):
    root = copy_real(tmp_path, mut, drop)
    with pytest.raises(ConfigError, match=match):
        load_templates(root, cfg)


def test_unreadable_or_empty_patch_file_is_config_error(cfg, tmp_path):
    root = copy_real(tmp_path)
    (root / "real_back_v1" / "digit_3.png").write_bytes(b"")
    with pytest.raises(ConfigError, match="digit_3.png"):
        load_templates(root, cfg)
    root2 = copy_real(tmp_path / "b")
    (root2 / "real_back_v1" / "digit_1.png").write_bytes(b"not a png")
    with pytest.raises(ConfigError, match="digit_1.png"):
        load_templates(root2, cfg)


# ---------------------------------------------------------------- §1.8b: field_rect と protected_rects の隙間
def test_protected_gap_below_required_is_config_error(cfg, tmp_path):
    need = float(cfg.mask.protected_min_gap_px) + float(cfg.mask.protected_clear_px)
    assert need == 9.0
    root = copy_real(tmp_path, lambda m: m.__setitem__("protected_rects", [[56.0, 52.0, 955.0, 295.0]]))  # 隙間 5px
    with pytest.raises(ConfigError, match=r"protected_rects\[0\]") as e:
        load_templates(root, cfg)
    msg = str(e.value)
    assert "5" in msg and "9" in msg and "下" in msg


def test_protected_gap_exactly_required_passes(cfg, tmp_path):
    root = copy_real(tmp_path, lambda m: m.__setitem__("protected_rects", [[56.0, 52.0, 955.0, 291.0]]))  # 隙間 9px
    tmpls, _ = load_templates(root, cfg)
    assert [t.id for t in tmpls] == ["real_back_v1"]


def test_protected_gap_follows_the_config_values(cfg):
    """9 を直書きしていない: mask 設定の和で決まる。"""
    meta = {"field_rect": [40.0, 300.0, 971.0, 597.0], "protected_rects": [[56.0, 52.0, 955.0, 288.0]]}
    _validate_protected_gap("t", meta, cfg)                                              # 隙間 12px ≥ 9
    wide = cfg.override("mask", protected_min_gap_px=10, protected_clear_px=3)           # 必要 13px > 12
    with pytest.raises(ConfigError, match="13"):
        _validate_protected_gap("t", meta, wide)


def test_overlapping_protected_rect_is_config_error(cfg):
    meta = {"field_rect": [40.0, 300.0, 971.0, 597.0], "protected_rects": [[56.0, 52.0, 955.0, 301.0]]}
    with pytest.raises(ConfigError, match=r"重な.*protected_rects\[0\]|protected_rects\[0\].*重な"):
        _validate_protected_gap("t", meta, cfg)


def test_facing_sides_are_measured_on_all_four_sides(cfg):
    ok = {"field_rect": [40.0, 300.0, 971.0, 597.0], "protected_rects": [[0.0, 400.0, 31.0, 500.0]]}   # 隙間 9
    _validate_protected_gap("t", ok, cfg)
    ng = {"field_rect": [40.0, 300.0, 971.0, 597.0], "protected_rects": [[0.0, 400.0, 32.0, 500.0]]}   # 隙間 8
    with pytest.raises(ConfigError, match=r"protected_rects\[0\].*右"):
        _validate_protected_gap("t", ng, cfg)
    right = {"field_rect": [40.0, 300.0, 900.0, 597.0], "protected_rects": [[905.0, 400.0, 990.0, 500.0]]}  # 隙間 5
    with pytest.raises(ConfigError, match=r"protected_rects\[0\].*左"):
        _validate_protected_gap("t", right, cfg)
    above_ng = {"field_rect": [40.0, 300.0, 971.0, 597.0], "protected_rects": [[56.0, 52.0, 955.0, 295.0]]}
    with pytest.raises(ConfigError, match=r"protected_rects\[0\].*下"):
        _validate_protected_gap("t", above_ng, cfg)
    below = {"field_rect": [40.0, 300.0, 971.0, 560.0], "protected_rects": [[56.0, 565.0, 955.0, 630.0]]}
    with pytest.raises(ConfigError, match=r"protected_rects\[0\].*上"):
        _validate_protected_gap("t", below, cfg)


def test_non_facing_protected_rects_are_not_measured(cfg):
    """向かい合う辺だけを測る: 範囲が重ならない（斜め）や遠い矩形は隙間 0 でも対象外。"""
    f = [40.0, 300.0, 971.0, 597.0]
    # 左右は y 範囲が重なる場合のみ: 欄の左上の外（y が重ならない）。上下は x 範囲が重なる場合のみ
    _validate_protected_gap("t", {"field_rect": f, "protected_rects": [[0.0, 0.0, 40.0, 299.0]]}, cfg)
    _validate_protected_gap("t", {"field_rect": f, "protected_rects": [[0.0, 0.0, 39.0, 300.0]]}, cfg)
    _validate_protected_gap("t", {"field_rect": f, "protected_rects": []}, cfg)


# ---------------------------------------------------------------- config.toml の範囲検査
CFG_TEXT = (ROOT / "config.toml").read_text(encoding="utf-8")


def cfg_with(tmp_path, old, new):
    assert old in CFG_TEXT, old
    p = tmp_path / "config.toml"
    p.write_text(CFG_TEXT.replace(old, new, 1), encoding="utf-8")
    return p


def test_new_config_keys_have_the_provisional_values(cfg):
    f, v = cfg.field, cfg.verify
    assert (f.s_digit, f.digit_pitch_tol, f.digit_xshift_h, f.digit_gap_ratio_max, f.digit_nms_px, f.digit_max_peaks) \
        == (0.6, 0.15, 0.5, 1.2, 7, 120)
    assert v.digit_absent_max == 0.3


@pytest.mark.parametrize("old, new", [
    ("s_digit = 0.6", "s_digit = 0"), ("s_digit = 0.6", "s_digit = 1.01"), ("s_digit = 0.6", "s_digit = -0.1"),
    ("digit_pitch_tol = 0.15", "digit_pitch_tol = 0"), ("digit_pitch_tol = 0.15", "digit_pitch_tol = 1"),
    ("digit_xshift_h = 0.5", "digit_xshift_h = 0"), ("digit_xshift_h = 0.5", "digit_xshift_h = -1"),
    ("digit_gap_ratio_max = 1.2", "digit_gap_ratio_max = 0.99"),
    ("digit_nms_px = 7", "digit_nms_px = 0"),
    ("digit_max_peaks = 120", "digit_max_peaks = 0"),
    ("digit_absent_max = 0.3", "digit_absent_max = 0"), ("digit_absent_max = 0.3", "digit_absent_max = 1"),
])
def test_new_config_ranges_are_validated(tmp_path, old, new):
    with pytest.raises(ConfigError):
        load_config(cfg_with(tmp_path, old, new))


@pytest.mark.parametrize("key", ["s_digit", "digit_pitch_tol", "digit_xshift_h", "digit_gap_ratio_max",
                                 "digit_nms_px", "digit_max_peaks", "digit_absent_max"])
def test_new_config_keys_are_required(tmp_path, key):
    lines = [ln for ln in CFG_TEXT.splitlines() if not ln.lstrip().startswith(key + " ")]
    assert len(lines) == len(CFG_TEXT.splitlines()) - 1
    p = tmp_path / "config.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(p)


# ---------------------------------------------------------------- tools/cut_digit_patches.py
def load_tool():
    spec = importlib.util.spec_from_file_location("cut_digit_patches", ROOT / "tools" / "cut_digit_patches.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cut_tool_writes_patches_and_reports_centers(synth, tmp_path, capsys):
    tool = load_tool()
    src = tmp_path / "card.png"
    ok, buf = cv2.imencode(".png", synth["gray"])
    src.write_bytes(buf.tobytes())
    out = tmp_path / "out"
    rc = tool.main(["--in", str(src), "--boxes", json.dumps([list(b) for b in synth["boxes"]]),
                    "--out", str(out), "--margin", str(MARGIN)])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["centers"] == synth["centers"] and rep["pitch"] == pytest.approx(60.0)
    for i, p in enumerate(synth["patches"], 1):
        got = cv2.imdecode(np.fromfile(str(out / f"digit_{i}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(got, p)


def test_cut_tool_never_overwrites(synth, tmp_path, capsys):
    tool = load_tool()
    src = tmp_path / "card.png"
    ok, buf = cv2.imencode(".png", synth["gray"])
    src.write_bytes(buf.tobytes())
    out = tmp_path / "out"
    out.mkdir()
    (out / "digit_2.png").write_bytes(b"keep me")
    rc = tool.main(["--in", str(src), "--boxes", json.dumps([list(b) for b in synth["boxes"]]), "--out", str(out)])
    assert rc != 0
    assert (out / "digit_2.png").read_bytes() == b"keep me"
    assert not (out / "digit_1.png").exists()           # 1枚も書かない


def test_cut_tool_reproduces_the_committed_real_patches(tmp_path, capsys):
    """既存の digit_*.png と同じバイト列になる（出力先は別ディレクトリ。既存は触らない）。

    切り出し元の c1（ECC 後）画像はリポジトリに無いので、コミット済みパッチを実測位置（外接矩形＋5px）に
    貼った無地のキャンバスを元画像にして、ツールの切り出し（箱の解釈・余白・PNG 符号化）が
    バイト一致することを確認する。
    """
    import hashlib
    tool = load_tool()
    boxes = [[74, 363, 80, 381], [73, 392, 84, 411], [73, 422, 84, 441]]
    canvas = np.full((fx.NH, fx.NW), 235, np.uint8)
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        p = cv2.imdecode(np.fromfile(str(REAL_DIR / f"digit_{i}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
        canvas[y0 - 5:y1 + 6, x0 - 5:x1 + 6] = p
    src = tmp_path / "c1.png"
    ok, buf = cv2.imencode(".png", canvas)
    src.write_bytes(buf.tobytes())
    out = tmp_path / "out"
    assert tool.main(["--in", str(src), "--boxes", json.dumps(boxes), "--out", str(out), "--margin", "5"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["centers"] == [[77.5, 372.5], [79.0, 402.0], [79.0, 432.0]]
    for i in (1, 2, 3):
        assert hashlib.sha256((out / f"digit_{i}.png").read_bytes()).hexdigest() == \
            hashlib.sha256((REAL_DIR / f"digit_{i}.png").read_bytes()).hexdigest()
