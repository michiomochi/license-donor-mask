"""`run --diagnostic`（設計 §1.1a）: 各段の fail を記録するだけで先へ進み、全段のメトリクスをログに残す。

画像は全て tests/fixtures_gen.py の合成データ。診断は ok/ にも review/ にも何も書かず、status は常に review。
前段の結果が無くて計算できない段は値を推測せず {"skipped": "<前提段>"} を残す。
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import ok_files, reason_codes

import fixtures_gen as fx
from donor_mask import cli, imgio, pipeline

STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")


def write_input(tmp_path, fixture, name="SYNTHETIC_TEST_in"):
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    p = d / (name + fixture.ext)
    p.write_bytes(fixture.data)
    return p


def diagnose(tmp_path, fixture, cfg, tmpls, name="SYNTHETIC_TEST_in"):
    templates, tsha = tmpls
    p = write_input(tmp_path, fixture, name)
    out = tmp_path / "out"
    rec, halt = pipeline.diagnose_file(p, out, cfg, templates, tsha)
    assert not halt
    return rec, out, p


def assert_nothing_but_log(out, rec):
    """ok/ と review/ には何も作らない。書くのは out/logs/<job_id>.json だけ。"""
    assert ok_files(out) == []
    assert not (out / "review").exists()
    assert not (out / "ok" / ".staging").exists()
    assert [p.name for p in (out / "logs").iterdir()] == [f"{rec['job_id']}.json"]
    assert rec["outputs"] == {"masked": None, "masked_sha256": None, "preview": None, "review": None}


def test_diagnostic_on_a_passing_card_is_still_review_and_logs_every_stage(cfg, tmpls, tmp_path):
    f = fx.make({"seed": 1})
    rec, out, p = diagnose(tmp_path, f, cfg, tmpls)
    assert rec["mode"] == "diagnostic"
    assert rec["status"] == "review"                      # 常に review（全段 pass でも ok にしない）
    assert rec["reasons"] == []                           # 落ちた段が無いので理由は空
    assert set(rec["stages"]) == set(STAGES)
    for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        assert "skipped" not in rec["stages"][s], s
    assert all(rec["stages"]["S8"][k]["pass"] for k in ("V1", "V2", "V3", "V4", "V5", "V6"))
    assert rec["stages"]["S9"] == {"skipped": "書き出しは run のみ"}
    assert rec["job_id"] == imgio.sha256_bytes(p.read_bytes())[:16] == rec["input"]["sha256"][:16]
    assert rec["original_unchanged"] is True
    assert rec["geometry"] is not None and rec["geometry"]["mask_polygon"]
    assert_nothing_but_log(out, rec)
    on_disk = json.loads((out / "logs" / f"{rec['job_id']}.json").read_text(encoding="utf-8"))
    assert on_disk["mode"] == "diagnostic" and on_disk["status"] == "review"


def test_normal_run_logs_mode_run(cfg, tmpls, tmp_path):
    templates, tsha = tmpls
    p = write_input(tmp_path, fx.make({"seed": 1}))
    rec, _ = pipeline.process_file(p, tmp_path / "out", cfg, templates, tsha)
    assert rec["status"] == "ok" and rec["mode"] == "run"


def test_diagnostic_matches_the_normal_run_metrics(cfg, tmpls, tmp_path):
    """診断と通常 run は同じ段関数を使う。段のメトリクスは（S9 以外）一致する。"""
    templates, tsha = tmpls
    f = fx.make({"seed": 2, "angle": 5})
    p = write_input(tmp_path, f)
    run_rec, _ = pipeline.process_file(p, tmp_path / "o_run", cfg, templates, tsha)
    diag_rec, _ = pipeline.diagnose_file(p, tmp_path / "o_diag", cfg, templates, tsha)
    assert run_rec["status"] == "ok"
    for s in STAGES[:-1]:
        assert diag_rec["stages"][s] == run_rec["stages"][s], s
    assert diag_rec["geometry"] == run_rec["geometry"]


def test_diagnostic_goes_on_after_a_failed_stage(cfg, tmpls, tmp_path, monkeypatch):
    """S5 が落ちても S6/S7/S8 まで計算し、S5 の理由コードだけが reasons に残る。"""
    real = pipeline.s5_quality

    def bad_quality(aligned, tmpl, card_long, c):
        met, rs = real(aligned, tmpl, card_long, c)
        return met, rs + [{"code": "Q_BLUR", "metric": "laplacian_var", "value": 1.0, "threshold": 60.0}]

    monkeypatch.setattr(pipeline, "s5_quality", bad_quality)
    rec, out, _ = diagnose(tmp_path, fx.make({"seed": 1}), cfg, tmpls)
    assert rec["status"] == "review"
    assert reason_codes(rec) == ["Q_BLUR"] and rec["reasons"][0]["stage"] == "S5"
    for s in ("S6", "S7"):
        assert "skipped" not in rec["stages"][s]
    assert "area_ratio" in rec["stages"]["S7"]
    assert set(rec["stages"]["S8"]) >= {"V1", "V2", "V3", "V4", "V5", "V6"}
    assert_nothing_but_log(out, rec)


def test_diagnostic_keeps_reasons_of_every_failed_stage(cfg, tmpls, tmp_path, monkeypatch):
    real = pipeline.s5_quality

    def bad_quality(aligned, tmpl, card_long, c):
        met, rs = real(aligned, tmpl, card_long, c)
        return met, rs + [{"code": "Q_GLARE", "metric": "glare_ratio", "value": 0.5, "threshold": 0.02}]

    def bad_s6(aligned, tmpl, c):
        return None, {"F": {"found": False}}, [{"code": "F_FRAME_NOT_FOUND", "metric": "f_rule_support",
                                                "value": 0.1, "threshold": 0.85}]

    monkeypatch.setattr(pipeline, "s5_quality", bad_quality)
    monkeypatch.setattr(pipeline, "s6_locate", bad_s6)
    rec, out, _ = diagnose(tmp_path, fx.make({"seed": 1}), cfg, tmpls)
    assert reason_codes(rec) == ["Q_GLARE", "F_FRAME_NOT_FOUND"]
    assert [r["stage"] for r in rec["reasons"]] == ["S5", "S6"]
    # cues が無いので S7 は計算できない（値を推測しない）。S8・S9 も連鎖して skipped
    assert rec["stages"]["S7"] == {"skipped": "S6"}
    assert rec["stages"]["S8"] == {"skipped": "S7"}
    assert rec["stages"]["S9"] == {"skipped": "S8"}
    assert rec["geometry"] is None
    assert_nothing_but_log(out, rec)


def test_diagnostic_s7_failure_skips_only_s8(cfg, tmpls, tmp_path, monkeypatch):
    def bad_s7(aligned, tmpl, cues, c):
        return None, {"R": list(cues["R"])}, [{"code": "M_PROTECTED_CONFLICT", "metric": "pad_after_shrink",
                                              "value": -1.0, "threshold": 6}]

    monkeypatch.setattr(pipeline, "s7_mask", bad_s7)
    rec, out, _ = diagnose(tmp_path, fx.make({"seed": 1}), cfg, tmpls)
    assert reason_codes(rec) == ["M_PROTECTED_CONFLICT"] and rec["reasons"][0]["stage"] == "S7"
    assert rec["stages"]["S6"].get("F") is not None and "skipped" not in rec["stages"]["S6"]
    assert rec["stages"]["S8"] == {"skipped": "S7"} and rec["stages"]["S9"] == {"skipped": "S8"}
    assert_nothing_but_log(out, rec)


def test_diagnostic_without_card_candidates_skips_s2_to_s7(cfg, tmpls, tmp_path):
    ok, buf = cv2.imencode(".png", np.full((700, 1000, 3), 128, np.uint8))
    assert ok
    p = tmp_path / "in" / "SYNTHETIC_TEST_blank.png"
    p.parent.mkdir()
    p.write_bytes(buf.tobytes())
    templates, tsha = tmpls
    rec, halt = pipeline.diagnose_file(p, tmp_path / "out", cfg, templates, tsha)
    assert not halt and rec["status"] == "review"
    assert "C_NO_CARD" in reason_codes(rec) and rec["reasons"][0]["stage"] == "S1"
    for s in ("S2", "S3", "S4", "S5", "S6", "S7"):
        assert rec["stages"][s] == {"skipped": "S1"}, s
    assert rec["stages"]["S8"] == {"skipped": "S7"} and rec["stages"]["S9"] == {"skipped": "S8"}
    assert_nothing_but_log(tmp_path / "out", rec)


def test_diagnostic_on_front_card_records_s3_and_skips_the_rest(cfg, tmpls, tmp_path):
    """表面は S3 で O_FRONT_SIDE。positive 様式ではないので S4 以降は計算しない（skipped: S3）。"""
    rec, out, _ = diagnose(tmp_path, fx.make({"kind": "front", "seed": 2}), cfg, tmpls)
    assert rec["status"] == "review" and "O_FRONT_SIDE" in reason_codes(rec)
    assert rec["stages"]["S3"]["best"] is not None
    for s in ("S4", "S5", "S6", "S7"):
        assert rec["stages"][s] == {"skipped": "S3"}, s
    assert_nothing_but_log(out, rec)


def test_diagnostic_exception_in_a_stage_is_recorded_and_dependents_skipped(cfg, tmpls, tmp_path, monkeypatch):
    def boom(aligned, tmpl, c):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "s6_locate", boom)
    rec, out, _ = diagnose(tmp_path, fx.make({"seed": 1}), cfg, tmpls)
    assert reason_codes(rec) == ["X_EXCEPTION"] and rec["reasons"][0]["stage"] == "S6"
    assert rec["stages"]["S7"] == {"skipped": "S6"}


def test_diagnostic_does_not_overwrite_an_existing_log(cfg, tmpls, tmp_path):
    templates, tsha = tmpls
    p = write_input(tmp_path, fx.make({"seed": 1}))
    out = tmp_path / "out"
    r1, _ = pipeline.diagnose_file(p, out, cfg, templates, tsha)
    before = (out / "logs" / f"{r1['job_id']}.json").read_bytes()
    r2, _ = pipeline.diagnose_file(p, out, cfg, templates, tsha)
    assert r2["status"] == "error" and reason_codes(r2) == ["X_OUTPUT_EXISTS"]
    assert r2["primary_reason"] == "X_OUTPUT_EXISTS"
    assert (out / "logs" / f"{r1['job_id']}.json").read_bytes() == before


# ---------------------------------------------------------------- CLI
def test_cli_diagnostic_excludes_shadow(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--diagnostic", "--shadow", "--in", str(tmp_path), "--out", str(tmp_path / "o")])
    assert e.value.code == 2


def test_cli_diagnostic_prints_one_line_per_file_and_exits_3(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    (d / "SYNTHETIC_TEST_ok.jpg").write_bytes(fx.make({"seed": 1}).data)
    (d / "SYNTHETIC_TEST_front.png").write_bytes(fx.make({"kind": "front", "seed": 2, "jpeg": None}).data)
    out = tmp_path / "out"
    rc = cli.main(["run", "--diagnostic", "--in", str(d), "--out", str(out)])
    assert rc == 3                                          # review のみ（ok は無い）
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[diag]")]
    assert len(lines) == 2
    ok_line = next(l for l in lines if "SYNTHETIC_TEST_ok.jpg" in l)
    assert ok_line.endswith("-> review")
    for s in ("S3=pass", "S4=pass", "S5=pass", "S6=pass", "S7=pass", "S8=pass"):
        assert s in ok_line, ok_line
    front_line = next(l for l in lines if "SYNTHETIC_TEST_front.png" in l)
    assert "S3=fail" in front_line and "S4=skip" in front_line
    # 落ちた段の理由コードを通常 run の行と同じ形式（カンマ区切り）で末尾に付ける
    assert "-> review  reasons: " in front_line, front_line
    codes = front_line.split("-> review  reasons: ")[1].split(",")
    jobs = [json.loads(l.read_text()) for l in (out / "logs").glob("*.json") if not l.name.startswith("batch_")]
    front_rec = next(j for j in jobs if j["input"]["path"].endswith("front.png"))
    assert codes == list(dict.fromkeys(r["code"] for r in front_rec["reasons"])) and codes
    assert ok_files(out) == [] and not (out / "review").exists()
    logs = sorted((out / "logs").glob("*.json"))
    jobs = [json.loads(l.read_text()) for l in logs if not l.name.startswith("batch_")]
    assert len(jobs) == 2 and all(j["mode"] == "diagnostic" and j["status"] == "review" for j in jobs)
    batch = json.loads(next(l for l in logs if l.name.startswith("batch_")).read_text())
    assert batch["counts"] == {"ok": 0, "review": 2, "error": 0} and batch["mode"] == "diagnostic"
