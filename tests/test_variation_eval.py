"""撮影バリエーション（背景・傾き・指・縮小・ブレ）の生成と、ok 率・誤検出の評価（tests/variation_gen.py / variation_eval.py）。

全て合成見本（SYNTHETIC_TEST）。評価の網は固定で、結果を見て調整しない。誤検出（塗り位置ずれ・カード以外を塗る 他）は
「ok になったのに塗りが正しくない」件で、ここでは 0 であることを要求する（保護領域への食い込みだけは下の注記のとおり別枠）。
"""
import numpy as np
import pytest

import fixtures_gen as fx
import variation_eval as ve
import variation_gen as vg

HARD_KINDS = ("misplaced", "outside_card", "overpaint", "not_fill_colour")  # これが 1 件でもあれば失敗
PROTECTED_BOUND_PX = 4.0  # 既知の所見（README 参照）: 縮小したカードで保護領域へ ~3px 食い込む。実測の最大 2.87px。悪化の検知用


@pytest.fixture(scope="module")
def sample():
    return vg.synthetic_sample()


# ---------------------------------------------------------------- 生成関数

def test_variation_is_deterministic(sample):
    c = {"bg": "wood", "angle": 7, "persp": 0.04, "finger": "left", "shrink": 0.6, "blur": 1.0, "seed": 3}
    a, b = vg.make_variation(sample, c), vg.make_variation(sample, c)
    assert a.data == b.data and np.allclose(a.quad, b.quad) and np.allclose(a.gt_field, b.gt_field)
    assert vg.make_variation(sample, dict(c, seed=4)).data != a.data


def test_unknown_parameter_is_an_error(sample):
    with pytest.raises(KeyError):
        vg.make_variation(sample, {"bokeh": 3})


def test_grid_is_fixed_and_covers_all_axes():
    g = vg.build_grid()
    assert set(g) == {"background", "tilt", "finger", "shrink", "blur", "combo"}
    assert g == vg.build_grid()  # 乱択（combo）も固定シードで同じ
    assert sum(len(v) for v in g.values()) == 70
    for cases in g.values():
        for c in cases:
            assert set(c) <= set(vg.DEFAULTS)


@pytest.mark.parametrize("bg", vg.BACKGROUNDS)
def test_every_background_renders(bg):
    img = vg.make_background(bg, 320, 200, 1)
    assert img.shape == (200, 320, 3) and img.dtype == np.uint8


def test_ground_truth_follows_the_geometry(sample):
    """GT の欄の 4 角が、カード四角形の中にあり、正規化座標→画像の射影で見本の欄に対応する。"""
    f = vg.make_variation(sample, {"angle": 12, "persp": 0.05, "bg": "plain", "jpeg": None})
    h, w = f.img.shape[:2]
    assert (f.quad >= 0).all() and (f.quad[:, 0] < w).all() and (f.quad[:, 1] < h).all()
    import cv2
    for p in f.gt_field:
        assert cv2.pointPolygonTest(f.quad.astype(np.float32), (float(p[0]), float(p[1])), False) > 0
    # 欄の中心を見本の欄中心へ戻したときの射影: カード 4 角 → 正規化 4 角 で作った H で欄の GT が元の矩形に戻る
    H = cv2.getPerspectiveTransform(f.quad.astype(np.float32), np.array([[0, 0], [NW := fx.NW, 0], [NW, fx.NH], [0, fx.NH]], np.float32))
    back = cv2.perspectiveTransform(f.gt_field.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    x0, y0, x1, y1 = sample.field_rect
    assert np.allclose(back, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], atol=1.0)


def test_shrink_scales_ground_truth(sample):
    a = vg.make_variation(sample, {"bg": "plain", "jpeg": None, "shrink": 1.0})
    b = vg.make_variation(sample, {"bg": "plain", "jpeg": None, "shrink": 0.5})
    assert abs(b.img.shape[1] - a.img.shape[1] * 0.5) <= 1
    assert np.allclose((a.gt_field + 0.5) * 0.5 - 0.5, b.gt_field, atol=0.5)


def test_finger_changes_pixels_near_the_edge_only(sample):
    a = vg.make_variation(sample, {"bg": "plain", "jpeg": None})
    b = vg.make_variation(sample, {"bg": "plain", "jpeg": None, "finger": "bottom"})
    diff = np.any(a.img != b.img, axis=2)
    assert diff.any()
    ys = np.nonzero(diff.any(axis=1))[0]
    assert ys.min() > np.mean(a.quad[:, 1])  # 下から差し込む: カード中心より上は変わらない
    c = vg.make_variation(sample, {"bg": "plain", "jpeg": None, "finger": "field"})
    assert np.any(a.img != c.img, axis=2)[fx.field_mask(a.img.shape, a, erode=2)].any()  # field は欄を隠す


def test_blur_lowers_sharpness(sample):
    import cv2
    v = lambda c: cv2.Laplacian(cv2.cvtColor(vg.make_variation(sample, dict(c, jpeg=None)).img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    base = v({})
    assert v({"blur": 2.0}) < base and v({"blur": 15, "blur_kind": "motion"}) < base


# ---------------------------------------------------------------- 評価関数（誤検出の判定そのものの検査）

def _ok_rec(tmp_path, fixture, edit):
    """原本を decode → edit(out) で加工 → PNG に書いた『ok のつもりの出力』の rec。"""
    import cv2
    from donor_mask import imgio
    orig, _ = imgio.decode(fixture.data, 10**9)
    out = orig.copy()
    edit(out, fixture)
    p = tmp_path / "m.png"
    cv2.imwrite(str(p), out)
    return {"outputs": {"masked": str(p)}}


def _fill_poly(img, pts, grow=0):
    import cv2
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    if grow:
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    img[m.astype(bool)] = 0


def test_evaluator_accepts_a_correct_fill(sample, tmp_path):
    f = vg.make_variation(sample, {"bg": "plain", "jpeg": None})
    rec = _ok_rec(tmp_path, f, lambda out, fx_: _fill_poly(out, fx_.gt_field, grow=4))
    assert ve.false_ok_kinds(f, rec) == []


def test_evaluator_flags_misplaced_fill(sample, tmp_path):
    f = vg.make_variation(sample, {"bg": "plain", "jpeg": None})
    shifted = lambda out, fx_: _fill_poly(out, fx_.gt_field + np.array([0, 60.0]), grow=2)  # 欄が見えている
    assert "misplaced" in ve.false_ok_kinds(f, _ok_rec(tmp_path, f, shifted))


def test_evaluator_flags_painting_outside_the_card(sample, tmp_path):
    f = vg.make_variation(sample, {"bg": "plain", "jpeg": None})

    def edit(out, fx_):
        _fill_poly(out, fx_.gt_field, grow=4)
        out[2:40, 2:40] = 0  # 背景の角を塗る
    assert "outside_card" in ve.false_ok_kinds(f, _ok_rec(tmp_path, f, edit))


def test_evaluator_flags_paint_into_protected_and_overpaint(sample, tmp_path):
    f = vg.make_variation(sample, {"bg": "plain", "jpeg": None})

    def into_protected(out, fx_):
        _fill_poly(out, fx_.gt_field, grow=4)
        _fill_poly(out, fx_.gt_protected[0] * 0.5 + np.mean(fx_.gt_protected[0], axis=0) * 0.5)
    assert "protected" in ve.false_ok_kinds(f, _ok_rec(tmp_path, f, into_protected))

    def overpaint(out, fx_):
        _fill_poly(out, fx_.gt_field, grow=80)
    assert "overpaint" in ve.false_ok_kinds(f, _ok_rec(tmp_path, f, overpaint))


def test_evaluator_flags_non_fill_colour_change(sample, tmp_path):
    f = vg.make_variation(sample, {"bg": "plain", "jpeg": None})

    def edit(out, fx_):
        _fill_poly(out, fx_.gt_field, grow=4)
        x0, y0, x1, y1 = [int(v) for v in np.r_[fx_.gt_field.min(axis=0), fx_.gt_field.max(axis=0)]]
        out[y0:y0 + 6, x0:x0 + 6] = (10, 20, 30)  # 黒でない変更
    assert "not_fill_colour" in ve.false_ok_kinds(f, _ok_rec(tmp_path, f, edit))


# ---------------------------------------------------------------- 本番経路での評価（固定の網の全件）

@pytest.fixture(scope="module")
def summary(cfg, tmpls, sample):
    res = ve.evaluate_grid([sample], vg.build_grid(), cfg, tmpls)
    s = ve.summarize(res)
    print("\n" + ve.format_text(s))
    return s


def test_grid_size_and_status_accounting(summary):
    t = summary["total"]
    assert t["n"] == 70 and t["ok"] + t["review"] + t["error"] == t["n"]
    assert t["error"] == 0  # X_EXCEPTION などの内部エラーは 0


def test_no_false_ok_of_hard_kinds(summary):
    """塗り位置ずれ・カード以外を塗る・塗りすぎ・塗り色以外の変更。ok になった件に 1 件も無いこと。"""
    kinds = {}
    for a in summary["axes"].values():
        for k, v in a["false_ok_kinds"].items():
            kinds[k] = kinds.get(k, 0) + v
    assert {k: v for k, v in kinds.items() if k in HARD_KINDS} == {}, summary["axes"]


def test_protected_intrusion_is_bounded(summary):
    """既知の所見: 縮小したカード（長辺 ~900px 以下）や大きな傾き（-30°）で、塗りが保護領域へ数 px 食い込む（'protected' の誤検出が
    実測 6/47 件）。今より悪化しないことだけを見る。疑われる原因は本番側（元画像での grow 膨張は px 固定、保護領域との隙間は
    縮小で縮む。未検証の推測）で、この評価の対象外（README の未解決事項）。"""
    assert summary["total"]["max_protected_depth_px"] <= PROTECTED_BOUND_PX


def test_ok_rate_does_not_regress(summary):
    """回帰の下限（目標値ではない）。実測 0.671（47/70）。軸別: tilt は全件 ok、finger/blur/combo は review が出る＝取りこぼしの側。"""
    assert summary["total"]["ok_rate"] >= 0.6
    assert summary["axes"]["tilt"]["ok_rate"] >= 0.9
