"""aggregate: run のログから件数と分布だけを出す。画像・パス・job_id・ファイル名を出さないこと、数え方が定義どおりなこと。

ログは手で組んだ合成 dict（SYNTHETIC_TEST）と、合成画像を run して得たログの両方を使う。実在の画像は使わない。
"""
import json

import fixtures_gen as fx
from donor_mask import aggregate, cli


def _rec(status, mode="run", reasons=(), primary=None, zncc=None, ecc=None, card_long=None, long_edge=None, job="j"):
    stages = {}
    if long_edge is not None:
        stages["S0"] = {"long_edge_px": long_edge}
    if zncc is not None:
        stages["S3"] = {"best": {"zncc": zncc}}
    if ecc is not None:
        stages["S4"] = {"correlation": ecc}
    if card_long is not None:
        stages["S5"] = {"card_long_edge_px": card_long}
    return {"schema_version": 1, "mode": mode, "job_id": job, "input": {"path": "/secret/name_taro.jpg"},
            "status": status, "primary_reason": primary, "stages": stages,
            "reasons": [{"code": c, "stage": "S1", "metric": "m", "value": "secret-value", "threshold": None}
                        for c in reasons]}


def _write_logs(d, recs):
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(recs):
        (d / f"{i:04d}.json").write_text(json.dumps(r), encoding="utf-8")


RECS = [
    _rec("ok", zncc=0.80, ecc=0.90, card_long=1000, long_edge=2000),
    _rec("ok", zncc=0.70, ecc=0.80, card_long=900, long_edge=1800),
    _rec("review", reasons=["C_NO_CARD", "C_ASPECT", "C_ASPECT"], primary="C_ASPECT", long_edge=1500),
    _rec("review", reasons=["Q_LOW_RES"], primary="Q_LOW_RES", zncc=0.60, ecc=0.70, card_long=300, long_edge=900),
    _rec("review", reasons=["O_LOW_SCORE", "Q_LOW_RES"], primary="O_LOW_SCORE", zncc=0.40, long_edge=1000),
    _rec("error", reasons=["X_IO"], primary="X_IO"),
]


def test_counts_status_and_reasons():
    a = aggregate.aggregate(RECS)
    assert a["n_logs"] == 6
    assert a["status"] == {"ok": 2, "review": 3, "error": 1}
    assert a["ok_rate"] == round(2 / 6, 4)
    # 同じコードが1件に複数あっても1件1回
    assert a["reason_counts"] == {"Q_LOW_RES": 2, "C_ASPECT": 1, "C_NO_CARD": 1, "O_LOW_SCORE": 1, "X_IO": 1}
    assert a["primary_reason_counts"]["(none)"] == 2
    assert a["primary_reason_counts"]["Q_LOW_RES"] == 1


def test_distributions_use_only_logs_that_reached_the_stage():
    m = aggregate.aggregate(RECS)["metrics"]
    z = m["zncc"]["all"]
    assert z["n"] == 4 and z["min"] == 0.4 and z["max"] == 0.8
    assert m["zncc"]["ok"]["n"] == 2 and m["zncc"]["review"]["n"] == 2
    assert "error" not in m["zncc"]  # 値のある件が無い status は出さない
    assert m["ecc_correlation"]["all"]["n"] == 3
    assert m["card_long_edge_px"]["all"]["n"] == 3 and m["card_long_edge_px"]["all"]["min"] == 300
    assert m["input_long_edge_px"]["all"]["n"] == 5
    h = m["card_long_edge_px"]["all"]["hist"]
    assert h["200-400"] == 1 and h["800-1200"] == 2 and sum(h.values()) == 3
    # ZNCC のヒストグラムは固定幅 0.1
    hz = m["zncc"]["all"]["hist"]
    assert hz["0.4-0.5"] == 1 and hz["0.8-0.9"] == 1 and sum(hz.values()) == 4


def test_non_numeric_and_missing_values_are_ignored():
    r = _rec("ok", zncc=0.7)
    r["stages"]["S4"] = {"correlation": "nan"}
    r["stages"]["S5"] = {"card_long_edge_px": True}
    r["stages"]["S0"] = {"long_edge_px": float("inf")}
    m = aggregate.aggregate([r])["metrics"]
    assert m["zncc"]["all"]["n"] == 1
    assert m["ecc_correlation"]["all"]["n"] == 0
    assert m["card_long_edge_px"]["all"]["n"] == 0 and m["input_long_edge_px"]["all"]["n"] == 0


def test_mode_filter():
    recs = RECS + [_rec("review", mode="diagnostic", reasons=["C_NO_CARD"], primary="C_NO_CARD")]
    assert aggregate.aggregate(recs)["n_logs"] == 7
    assert aggregate.aggregate(recs, "run")["n_logs"] == 6
    assert aggregate.aggregate(recs, "diagnostic")["n_logs"] == 1
    assert "diagnostic" in aggregate.format_text(aggregate.aggregate(recs))


def test_batch_logs_and_garbage_are_skipped(tmp_path):
    d = tmp_path / "out" / "logs"
    _write_logs(d, RECS)
    (d / "batch_abcd1234_efgh5678.json").write_text(json.dumps({"counts": {"ok": 9}, "jobs": [], "status": "ok"}))
    (d / "broken.json").write_text("{not json")
    (d / "list.json").write_text("[]")
    (d / "notes.txt").write_text("x")
    files = aggregate.find_log_files(tmp_path / "out")  # out を渡しても logs/ を見る
    assert files == aggregate.find_log_files(d)
    recs, skipped = aggregate.load_logs(files)
    assert len(recs) == 6 and skipped == 2  # broken.json と list.json（batch_ は最初から対象外）


def test_cli_prints_no_paths_job_ids_or_reason_values(tmp_path, capsys):
    d = tmp_path / "out" / "logs"
    _write_logs(d, [_rec("review", reasons=["C_NO_CARD"], primary="C_NO_CARD", job="deadbeefdeadbeef", zncc=0.5)] + RECS)
    for extra in ([], ["--json"]):
        assert cli.main(["aggregate", "--logs", str(tmp_path / "out")] + extra) == 0
        out = capsys.readouterr().out
        for leak in ("secret", "name_taro", "deadbeefdeadbeef", str(tmp_path), ".json"):
            assert leak not in out, leak
    assert cli.main(["aggregate", "--logs", str(tmp_path / "out"), "--json", "--mode", "run"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["n_logs"] == 7 and j["status"]["ok"] == 2


def test_cli_errors(tmp_path, capsys):
    assert cli.main(["aggregate", "--logs", str(tmp_path / "nope")]) == 1
    (tmp_path / "empty").mkdir()
    assert cli.main(["aggregate", "--logs", str(tmp_path / "empty")]) == 1
    capsys.readouterr()


def test_end_to_end_on_logs_written_by_run(tmp_path, capsys):
    """合成画像 2 枚（ok になる板 / カードなし）を run → そのログを aggregate。数え方が run の実ログで成り立つ。"""
    d = tmp_path / "in"
    d.mkdir()
    (d / "SYNTHETIC_TEST_ok.png").write_bytes(fx.make({"angle": 3, "margin": 0.1, "jpeg": None, "seed": 1}).data)
    (d / "SYNTHETIC_TEST_nocard.png").write_bytes(fx.make({"no_card": True, "jpeg": None, "seed": 2}).data)
    out = tmp_path / "out"
    assert cli.main(["run", "--in", str(d), "--out", str(out)]) == 3
    capsys.readouterr()
    assert cli.main(["aggregate", "--logs", str(out), "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["n_logs"] == 2 and j["status"] == {"ok": 1, "review": 1, "error": 0}
    assert j["reason_counts"].get("C_NO_CARD") == 1
    assert j["metrics"]["zncc"]["all"]["n"] == 1 and j["metrics"]["ecc_correlation"]["all"]["n"] == 1
    assert j["metrics"]["card_long_edge_px"]["ok"]["n"] == 1
    assert j["metrics"]["input_long_edge_px"]["all"]["n"] == 2
    txt = json.dumps(j)
    assert "SYNTHETIC_TEST" not in txt and str(tmp_path) not in txt
