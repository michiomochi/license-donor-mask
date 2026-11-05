"""verify.py の単体テスト。マスク処理とは無関係に作った画像・幾何を当て、それぞれ正しく落ちること。"""
import ast
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import ROOT, run_fixture

import fixtures_gen as fx
from donor_mask import imgio, verify
from donor_mask.geom import apply_h, rect_poly


@pytest.fixture(scope="module")
def ok_case(cfg, tmpls, tmp_path_factory):
    f = fx.make({"angle": 4, "seed": 21})
    tp = tmp_path_factory.mktemp("verify")
    rec, od, p = run_fixture(tp, f, cfg, tmpls)
    assert rec["status"] == "ok"
    orig, _ = imgio.decode(f.data, 10**9)
    masked = imgio.read_array(Path(rec["outputs"]["masked"]))
    tmpl = next(t for t in tmpls[0] if t.id == rec["geometry"]["template_id"])
    before = imgio.file_state(p)
    return {"rec": rec, "orig": orig, "masked": masked, "tmpl": tmpl, "geom": rec["geometry"], "path": p, "before": before}


def run_v(cfg, c, masked=None, geom=None):
    return verify.verify_all(orig_arr=c["orig"], masked_arr=c["masked"] if masked is None else masked,
                             geom=c["geom"] if geom is None else geom, tmpl=c["tmpl"], cfg=cfg, fill_bgr=(0, 0, 0),
                             orig_path=c["path"], orig_before=c["before"])


def test_verify_passes_on_the_real_output(cfg, ok_case):
    vr = run_v(cfg, ok_case)
    assert vr["pass"] and all(vr[k]["pass"] for k in ("V1", "V2", "V3", "V4", "V5", "V6"))


def test_non_uniform_fill_fails_v2(cfg, ok_case):
    m = ok_case["masked"].copy()
    mask = verify.make_mask(m.shape, ok_case["geom"]["mask_polygon"], int(cfg.mask.grow_px))
    ys, xs = np.nonzero(mask)
    m[ys[len(ys) // 2], xs[len(xs) // 2]] = (0, 0, 1)  # 1 画素だけ 1 段階違う
    vr = run_v(cfg, ok_case, masked=m)
    assert not vr["pass"] and not vr["V2"]["pass"] and vr["V2"]["code"] == "V_FILL_NOT_UNIFORM"


def test_one_pixel_changed_outside_fails_v3(cfg, ok_case):
    m = ok_case["masked"].copy()
    mask = verify.make_mask(m.shape, ok_case["geom"]["mask_polygon"], int(cfg.mask.grow_px))
    ys, xs = np.nonzero(~mask)
    y, x = ys[100], xs[100]
    m[y, x] = (m[y, x].astype(int) + 1).clip(0, 255).astype(np.uint8)
    vr = run_v(cfg, ok_case, masked=m)
    assert not vr["pass"] and vr["V3"]["code"] == "V_OUTSIDE_CHANGED" and vr["V3"]["metrics"]["diff_pixels_outside"] == 1


def test_shifted_mask_fails_v1_even_though_uniform_and_outside_intact(cfg, ok_case):
    """欄からずれた位置を塗った出力。V2/V3 は（ずれた矩形について）通るが、V1 が落とす。"""
    g = copy.deepcopy(ok_case["geom"])
    P = np.array(g["P_t2o"])
    r = g["mask_rect"]
    g["mask_rect"] = [r[0] + 30, r[1] + 30, r[2] + 30, r[3] + 30]
    g["mask_polygon"] = apply_h(P, rect_poly(g["mask_rect"])).tolist()
    mask = verify.make_mask(ok_case["orig"].shape, g["mask_polygon"], int(cfg.mask.grow_px))
    m = ok_case["orig"].copy()
    m[mask] = 0
    vr = run_v(cfg, ok_case, masked=m, geom=g)
    assert vr["V2"]["pass"] and vr["V3"]["pass"]
    assert not vr["V1"]["pass"] and vr["V1"]["code"] == "V_INCLUSION" and not vr["pass"]


def test_unmasked_output_leaves_trace_and_fails(cfg, ok_case):
    vr = run_v(cfg, ok_case, masked=ok_case["orig"].copy())
    assert not vr["pass"] and not vr["V2"]["pass"]


def test_v6_detects_changed_original(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"abc")
    before = imgio.file_state(p)
    assert verify.v6_original(p, before)["pass"]
    p.write_bytes(b"abd")
    assert not verify.v6_original(p, before)["pass"]
    p.write_bytes(b"abc")
    os.utime(p, ns=(before["mtime_ns"] + 10**9, before["mtime_ns"] + 10**9))
    r = verify.v6_original(p, before)
    assert not r["pass"] and r["metrics"]["sha256_same"] and not r["metrics"]["mtime_same"]


def test_verify_does_not_import_masking_modules():
    """verify.py は、マスクを作る側（mask / pipeline / orient）を import しない（二重化）。"""
    tree = ast.parse((ROOT / "donor_mask" / "verify.py").read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mods.add((n.module or "") if n.level == 0 else (n.module or ""))
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
    assert not ({"mask", "pipeline", "orient"} & mods), mods


def test_cli_verify_roundtrip(cfg, ok_case, capsys):
    from donor_mask import cli
    out_dir = Path(ok_case["rec"]["outputs"]["masked"]).parent.parent
    log = out_dir / "logs" / f"{ok_case['rec']['job_id']}.json"
    rc = cli.main(["verify", "--orig", str(ok_case["path"]), "--masked", ok_case["rec"]["outputs"]["masked"],
                   "--log", str(log)])
    assert rc == 0
    assert "verify: PASS" in capsys.readouterr().out
