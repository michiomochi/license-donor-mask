"""難しい背景（合成）: 明度がカードに近い色つきの背景・暗い背景に、傾けて置いたカードでも S1 が外形を取れること。

画像は全て SYNTHETIC_TEST（合成）。実在の免許証画像は使わない。ここで測るのは「候補が取れるか」と
「ok になったものが設計書 6.2 の不変条件を満たすか」で、特定の画像に合わせた閾値調整はしない。
"""
import numpy as np
import pytest
from conftest import assert_ok_invariants, run_fixture

import fixtures_gen as fx
from donor_mask.card import find_candidates

CASES = {}
for bg in ("clutter", "dark"):
    for a in (0, 8, -12, 20):
        CASES[f"{bg}_rot{a}"] = {"bg": bg, "angle": a, "margin": 0.12}
    CASES[f"{bg}_persp_small"] = {"bg": bg, "angle": 6, "persp": 0.05, "scale": 0.8, "margin": 0.1}


# 色つきまだら背景で 20° 傾けた1条件は GrabCut でも取れない（既知の限界。review になるだけで誤って塗ることはない）
S1_EXPECTED = [n for n in CASES if n != "clutter_rot20"]


@pytest.mark.parametrize("name", S1_EXPECTED)
def test_s1_finds_the_card_on_hard_background(cfg, tmpls, name):
    f = fx.make(dict(CASES[name], seed=7))
    res = find_candidates(f.img, cfg, tmpls[0])
    assert res["candidates"], (name, res["rejected"], res["grabcut_rejected"])
    q = np.array(res["candidates"][0]["quad"])
    diag = float(np.linalg.norm(f.quad[0] - f.quad[2]))
    assert np.abs(q - f.quad).max() < 0.03 * diag, (name, q.tolist(), f.quad.tolist())


@pytest.mark.parametrize("name", list(CASES))
def test_ok_on_hard_background_satisfies_invariants(cfg, tmpls, tmp_path, name):
    """ok にならない（review）のは許す（fail-closed）。ok になったなら不変条件を必ず満たす。"""
    f = fx.make(dict(CASES[name], seed=7))
    rec, _od, _p = run_fixture(tmp_path, f, cfg, tmpls, name=f"SYNTHETIC_TEST_{name}")
    if rec["status"] == "ok":
        assert_ok_invariants(f, rec)


def test_grabcut_is_the_fallback_that_finds_it(cfg, tmpls):
    """明度・色味の地図で取れない色つきまだら背景は、GrabCut の2地図の一致で取れる。無効にすると取れない（効果の確認）。"""
    f = fx.make({"bg": "clutter", "angle": 8, "margin": 0.12, "seed": 1})
    res = find_candidates(f.img, cfg, tmpls[0])
    assert res["candidates"] and res["candidates"][0]["maps"] == ["gc_a", "gc_b"]
    res_off = find_candidates(f.img, cfg.override("card", grabcut_fallback=0), tmpls[0])
    assert res_off["candidates"] == [] and res_off["grabcut_fallback"] is False


def test_grabcut_skips_low_colour_images(cfg, tmpls):
    """2色の絵のような色の少ない画像では GrabCut を使わない（数十秒かかって止まったように見えるのを防ぐ）。"""
    import time

    import cv2

    img = np.full((1200, 1200, 3), 235, np.uint8)
    cv2.rectangle(img, (0, 0), (1199, 1199), (30, 30, 30), 3)
    t0 = time.time()
    res = find_candidates(img, cfg, tmpls[0])
    assert res["candidates"] == [] and time.time() - t0 < 30
