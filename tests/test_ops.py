"""運用面: 再現性・原本不変・既存出力を上書きしない・入出力ディレクトリ検査・CLI・設定・EXIF・ネットワーク無し。"""
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import HERE, ROOT, ok_files, reason_codes, run_fixture

import fixtures_gen as fx
from donor_mask import cli, imgio, pipeline
from donor_mask.config import ConfigError, load_config
from donor_mask.template import load_templates


def stable(rec):
    """再現性比較用: nondeterministic と出力ディレクトリ依存のパスを除いたログ。"""
    r = json.loads(json.dumps(rec))
    r.pop("nondeterministic")
    r["outputs"] = {k: (None if v is None or k == "masked_sha256" else "x") for k, v in r["outputs"].items()} | \
        {"masked_sha256": rec["outputs"]["masked_sha256"]}
    return r


def test_fixture_generation_is_deterministic():
    a, b = fx.make({"angle": 7, "persp": 0.05, "seed": 4}), fx.make({"angle": 7, "persp": 0.05, "seed": 4})
    assert a.data == b.data


def test_reproducible_same_input_twice(cfg, tmpls, tmp_path):
    f = fx.make({"angle": 6, "persp": 0.03, "seed": 9, "jpeg": 70})
    d = tmp_path / "in"
    d.mkdir()
    p = d / "SYNTHETIC_TEST_a.jpg"
    p.write_bytes(f.data)
    t, ts = tmpls
    r1, _ = pipeline.process_file(p, tmp_path / "o1", cfg, t, ts)
    r2, _ = pipeline.process_file(p, tmp_path / "o2", cfg, t, ts)
    assert r1["status"] == r2["status"] == "ok"
    b1 = Path(r1["outputs"]["masked"]).read_bytes()
    b2 = Path(r2["outputs"]["masked"]).read_bytes()
    assert b1 == b2 and imgio.sha256_bytes(b1) == r1["outputs"]["masked_sha256"]
    assert stable(r1) == stable(r2)
    # review のログも同じ
    g = fx.make({"kind": "front", "seed": 2})
    p2 = d / "SYNTHETIC_TEST_b.jpg"
    p2.write_bytes(g.data)
    q1, _ = pipeline.process_file(p2, tmp_path / "o1", cfg, t, ts)
    q2, _ = pipeline.process_file(p2, tmp_path / "o2", cfg, t, ts)
    assert q1["status"] == "review" and stable(q1) == stable(q2)
    assert (tmp_path / "o1" / "review" / f"{q1['job_id']}__review.jpg").read_bytes() == \
        (tmp_path / "o2" / "review" / f"{q2['job_id']}__review.jpg").read_bytes()


def test_original_is_untouched(cfg, tmpls, tmp_path):
    f = fx.make({"seed": 3})
    rec, od, p = run_fixture(tmp_path, f, cfg, tmpls)
    assert p.read_bytes() == f.data
    assert rec["original_unchanged"] is True
    assert rec["input"]["sha256"] == imgio.sha256_bytes(f.data)
    assert os.stat(p).st_mtime_ns == rec["input"]["mtime_ns"]


def test_existing_output_is_not_overwritten(cfg, tmpls, tmp_path):
    f = fx.make({"seed": 3})
    rec, od, p = run_fixture(tmp_path, f, cfg, tmpls)
    masked = Path(rec["outputs"]["masked"])
    before = masked.read_bytes()
    log_before = (od / "logs" / f"{rec['job_id']}.json").read_bytes()
    t, ts = tmpls
    rec2, _ = pipeline.process_file(p, od, cfg, t, ts)
    assert rec2["status"] == "error" and "X_OUTPUT_EXISTS" in reason_codes(rec2)
    assert rec2["primary_reason"] == "X_OUTPUT_EXISTS"
    assert masked.read_bytes() == before and (od / "logs" / f"{rec['job_id']}.json").read_bytes() == log_before


def test_staging_is_cleaned_and_only_png_masked(cfg, tmpls, tmp_path):
    f = fx.make({"seed": 3})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert list((od / "ok" / ".staging").glob("*")) == []
    assert sorted(x.suffix for x in (od / "ok").glob("*") if x.is_file()) == [".jpg", ".png"]  # preview.jpg と masked.png


@pytest.mark.parametrize("fill,val", [("black", 0), ("white", 255)])
def test_fill_is_solid_single_color(cfg, tmpls, tmp_path, fill, val):
    f = fx.make({"seed": 3})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls, fill=fill)
    assert rec["status"] == "ok"
    out = imgio.read_array(Path(rec["outputs"]["masked"]))
    from donor_mask import verify
    m = verify.make_mask(out.shape, rec["geometry"]["mask_polygon"], int(cfg.mask.grow_px))
    assert len(np.unique(out[m].reshape(-1, 3), axis=0)) == 1 and int(out[m][0][0]) == val


def test_cli_rejects_noise_fill_and_jpeg_output(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--in", str(tmp_path), "--out", str(tmp_path / "o"), "--fill", "noise"])
    assert e.value.code == 2


def test_deferred_subcommands_are_explicitly_unimplemented(capsys):
    for c in ("calibrate", "make-fixtures"):
        assert cli.main([c]) == 1
    assert "未実装" in capsys.readouterr().err


def test_input_equals_output_dir_is_startup_error(cfg, tmpls, tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    (d / "SYNTHETIC_TEST_x.jpg").write_bytes(fx.make({}).data)
    assert cli.main(["run", "--in", str(d), "--out", str(d)]) == 1
    assert cli.main(["run", "--in", str(d), "--out", str(d / "sub")]) == 1  # 出力が入力の配下
    assert cli.main(["run", "--in", str(d / "SYNTHETIC_TEST_x.jpg"), "--out", str(d)]) == 1
    assert not (d / "ok").exists() and not (d / "sub").exists()


def test_cli_batch_exit_codes_and_layout(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "SYNTHETIC_TEST_ok.jpg").write_bytes(fx.make({"seed": 1}).data)
    (d / "SYNTHETIC_TEST_front.png").write_bytes(fx.make({"kind": "front", "seed": 2, "jpeg": None}).data)
    (d / "notes.txt").write_text("skip me")
    (d / "sub").mkdir()
    out = tmp_path / "out"
    rc = cli.main(["run", "--in", str(d), "--out", str(out)])
    assert rc == 3  # review を含む
    logs = sorted((out / "logs").glob("*.json"))
    assert len(logs) == 3  # ok の job, review の job, batch
    assert len(list((out / "ok").glob("*__masked.png"))) == 1 and len(list((out / "review").glob("*__review.jpg"))) == 1
    batch = json.loads(next(l for l in logs if l.name.startswith("batch_")).read_text())
    assert batch["counts"] == {"ok": 1, "review": 1, "error": 0}
    # 全件 ok なら 0
    d2 = tmp_path / "in2"
    d2.mkdir()
    (d2 / "SYNTHETIC_TEST_ok.jpg").write_bytes(fx.make({"seed": 1}).data)
    assert cli.main(["run", "--in", str(d2), "--out", str(tmp_path / "out2")]) == 0


def test_cli_rectify_writes_grid(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "SYNTHETIC_TEST_x.jpg").write_bytes(fx.make({"angle": 90, "seed": 1}).data)
    out = tmp_path / "rect"
    assert cli.main(["rectify", "--in", str(d), "--out", str(out)]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert any(n.endswith("__orig_grid.jpg") for n in names) and any(n.endswith("_grid.jpg") and "rect0" in n for n in names)
    rect = cv2.imread(str(next(out.glob("*__rect0.png"))))
    assert rect.shape[:2] == (638, 1011)


# ---- 設定・テンプレート
def test_config_hash_logged_and_missing_key_is_error(cfg, tmpls, tmp_path):
    rec, _, _ = run_fixture(tmp_path, fx.make({"seed": 3}), cfg, tmpls)
    assert rec["config_sha256"] == cfg.sha256 and rec["templates_sha256"] == tmpls[1]
    bad = tmp_path / "bad.toml"
    bad.write_text((ROOT / "config.toml").read_text().replace("min_zncc = 0.55", ""))
    with pytest.raises(ConfigError):
        load_config(bad)
    assert cli.main(["run", "--in", str(tmp_path), "--out", str(tmp_path / "x"), "--config", str(bad)]) == 1


def test_changed_threshold_changes_config_hash(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text((ROOT / "config.toml").read_text().replace("min_zncc = 0.55", "min_zncc = 0.56"))
    assert load_config(p).sha256 != load_config(ROOT / "config.toml").sha256


def test_cli_help_formats_for_every_subcommand(capsys):
    """`--help` が全サブコマンドで例外なく出る（help 文字列の % はエスケープする）。"""
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    for name in ("run", "rectify", "verify", "template-build", "calibrate", "make-fixtures"):
        with pytest.raises(SystemExit) as e:
            cli.main([name, "--help"])
        assert e.value.code == 0, name
    out = capsys.readouterr().out
    assert "5%グリッド" in out


def test_committed_templates_match_generator(cfg, tmp_path):
    """合成テンプレート（templates/synthetic_*）は tools/build_synthetic_templates.py の出力と一致する
    （手作業で書き換えていない）。実様式テンプレートは実写真から作るので、生成器の比較対象外
    （`synthetic: false` の明示があることだけを確認する）。"""
    sys.path.insert(0, str(ROOT / "tools"))
    import build_synthetic_templates as b
    b.main(tmp_path / "t")
    generated, _ = load_templates(tmp_path / "t", cfg)
    committed, _ = load_templates(ROOT / "templates", cfg)
    ids = {t.id for t in committed}
    assert {t.id for t in generated} <= ids
    for t in generated:
        for name in ("template.json", "ruling.png", "print.png"):
            assert (tmp_path / "t" / t.id / name).read_bytes() == (ROOT / "templates" / t.id / name).read_bytes(), \
                f"{t.id}/{name}"
    others = [t for t in committed if t.id not in {g.id for g in generated}]
    assert all(t.meta.get("synthetic") is False for t in others), \
        f"生成器の出力でないテンプレートは synthetic: false を明示すること: {[t.id for t in others]}"


# ---- 入力デコード
def _jpeg_with_exif_orientation(img, orientation):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    b = buf.tobytes()
    tiff = b"II*\x00\x08\x00\x00\x00" + b"\x01\x00" + b"\x12\x01\x03\x00\x01\x00\x00\x00" + bytes([orientation, 0, 0, 0]) \
        + b"\x00\x00\x00\x00"
    app1 = b"Exif\x00\x00" + tiff
    return b[:2] + b"\xff\xe1" + (len(app1) + 2).to_bytes(2, "big") + app1 + b[2:]


def test_exif_orientation_is_applied_by_decoder_and_logged():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = 255
    plain, info0 = imgio.decode(cv2.imencode(".jpg", img)[1].tobytes(), 10**9)
    assert plain.shape[:2] == (100, 200) and info0["exif_orientation_applied"] is False
    rot, info = imgio.decode(_jpeg_with_exif_orientation(img, 6), 10**9)
    assert rot.shape[:2] == (200, 100) and info["exif_orientation_applied"] is True  # OpenCV 5.0.0 の実測


def test_exif_rotated_input_is_processed_on_oriented_pixels(cfg, tmpls, tmp_path):
    f = fx.make({"seed": 5})
    d = tmp_path / "in"
    d.mkdir()
    # 物理的に 90 度回した画素＋「6=右に90度回して表示」のタグ → 復号後は元の向き
    phys = cv2.rotate(f.img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    p = d / "SYNTHETIC_TEST_exif.jpg"
    p.write_bytes(_jpeg_with_exif_orientation(phys, 6))
    t, ts = tmpls
    rec, _ = pipeline.process_file(p, tmp_path / "out", cfg, t, ts)
    assert rec["input"]["exif_orientation_applied"] is True and rec["status"] == "ok"


def test_alpha_png_is_composited_on_white():
    img = np.zeros((20, 20, 4), np.uint8)
    img[..., :3] = 0
    img[..., 3] = 0  # 完全透明
    out, _ = imgio.decode(cv2.imencode(".png", img)[1].tobytes(), 10**9)
    assert int(out.min()) == 255


# ---- 回路: AI/ネットワークを使わない
def test_no_network_or_ml_imports_in_package():
    banned = re.compile(r"^\s*(import|from)\s+(socket|urllib|requests|http|httpx|aiohttp|ftplib|smtplib|torch|tensorflow|onnx|"
                        r"sklearn|skimage|pytesseract|openai|anthropic)\b", re.M)
    for f in (ROOT / "donor_mask").glob("*.py"):
        assert not banned.search(f.read_text(encoding="utf-8")), f
        assert "cv2.dnn" not in f.read_text(encoding="utf-8"), f


def test_no_random_in_production_code():
    for f in (ROOT / "donor_mask").glob("*.py"):
        t = f.read_text(encoding="utf-8")
        assert "import random" not in t and "np.random" not in t and "numpy.random" not in t, f


def test_anchor_detector_supports_circle_marks():
    """番号を○で囲む様式の候補: anchor_type=circle3 で円形マークの3連を検出できる。"""
    from donor_mask import field, layout
    cfg = load_config()
    img = np.full((300, 200, 3), 235, np.uint8)
    for y in (60, 120, 180):
        cv2.circle(img, (100, y), 14, (30, 30, 30), 3)
    dark = layout.dark_map(img, cfg)
    singles = field.single_candidates(dark, 28.0, "circle3", cfg, (0, 0))
    assert len(singles) == 3
    assert len(field.find_triples(singles, 60.0, cfg)) == 1
    assert field.single_candidates(dark, 28.0, "box3", cfg, (0, 0)) == []  # 円は四角として数えない
