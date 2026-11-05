"""陽性（合成）: 想定レイアウトのカードを様々に撮影した体で処理し、ok になったものが設計書 6.2 を満たすこと。

画像は全て SYNTHETIC_TEST（合成）。固定グリッドの一部（一因子ずつ＋複合）を列挙する。生成は決定的（固定シード）。
"""
import pytest
from conftest import assert_ok_invariants, ok_files, run_fixture

import fixtures_gen as fx

CASES = {}
for a in (0, 5, -5, 15, -15, 90, 180, 270):
    CASES[f"rot{a}"] = {"angle": a}
for s, m in ((1.0, 0.1), (1.0, 0.3), (0.7, 0.0), (0.6, 0.0), (0.8, 0.1)):
    CASES[f"scale{s}_margin{m}"] = {"scale": s, "margin": m}
for p in (0.04, 0.08):
    for a in (0, 10):
        CASES[f"persp{p}_rot{a}"] = {"persp": p, "angle": a}
for g in (0.6, 1.6):
    CASES[f"gamma{g}"] = {"gamma": g}
for b in (1.0, 1.5):
    CASES[f"blur{b}"] = {"blur": b}
for j in (70, 95, None):
    CASES[f"jpeg{j}"] = {"jpeg": j}
for bg in ("plain", "texture", "desk_edge"):
    CASES[f"bg_{bg}"] = {"bg": bg}
CASES["upside_down_persp_jpeg70"] = {"angle": 180, "persp": 0.06, "jpeg": 70}
CASES["combo_rot-15_small_gamma1.6_blur1"] = {"angle": -15, "scale": 0.8, "margin": 0.1, "persp": 0.05,
                                              "gamma": 1.6, "jpeg": 70, "blur": 1.0}
CASES["stroke_outside_field"] = {"stroke": True}


@pytest.fixture(scope="module")
def results(cfg, tmpls, tmp_path_factory):
    out = {}
    for name, case in CASES.items():
        f = fx.make(dict(case, seed=len(out)))
        tp = tmp_path_factory.mktemp(name.replace(".", "_"))
        rec, od, _p = run_fixture(tp, f, cfg, tmpls, name=f"SYNTHETIC_TEST_{name}")
        out[name] = (f, rec, od)
    return out


def test_every_ok_satisfies_invariants(results):
    """ok になった全件が 包含率100%・欄外差分0・保護領域一致・塗りすぎ≤1.6。1件でも違反すればここで落ちる。"""
    n_ok = 0
    for name, (f, rec, _od) in results.items():
        if rec["status"] == "ok":
            n_ok += 1
            assert_ok_invariants(f, rec)
    assert n_ok >= 0.9 * len(results), f"自動 ok 率 {n_ok}/{len(results)}（参考値の目標は90%以上）"


@pytest.mark.parametrize("name", list(CASES))
def test_positive_case_is_ok(results, name):
    f, rec, od = results[name]
    assert rec["status"] == "ok", (name, rec["reasons"])
    assert rec["reasons"] == []
    assert all(rec["stages"]["S8"][k]["pass"] for k in ("V1", "V2", "V3", "V4", "V5", "V6"))
    names = ok_files(od)
    assert len(names) == 2 and any(n.endswith("__masked.png") for n in names) and any(n.endswith("__preview.jpg") for n in names)
    assert not (od / ".staging").exists() or not list((od / "ok" / ".staging").glob("*"))


def test_output_is_png_only_and_name_is_job_id(results):
    for name, (f, rec, od) in results.items():
        if rec["status"] != "ok":
            continue
        assert rec["outputs"]["masked"].endswith(f"{rec['job_id']}__masked.png")
        assert "SYNTHETIC_TEST" not in rec["outputs"]["masked"]  # 元ファイル名は出力名に使わない
        assert len(rec["job_id"]) == 16 and rec["job_id"] == rec["input"]["sha256"][:16]
