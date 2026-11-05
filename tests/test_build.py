"""`template-build`（設計書 5.4 C1）と、カードが枠いっぱいのときの `C_FRAME_FILLED` のテスト。

画像は全て tests/fixtures_gen.py の合成データ。実在の免許証・個人情報は使わない。
実写真での校正は別カード（templates/real_back_v1）で行う。
"""
import json

import cv2
import numpy as np
import pytest
from conftest import ROOT, reason_codes, run_fixture

import fixtures_gen as fx
from donor_mask import build, card, cli, pipeline
from donor_mask.card import find_candidates
from donor_mask.config import ConfigError, load_config
from donor_mask.geom import rect_area
from donor_mask.mask import base_pads, padded
from donor_mask.template import load_templates

SPEC = {
    "field_rect": list(fx.FIELD_RECT),
    "anchor_type": "box3",
    "anchor": {"side": fx.BOX_SIDE, "pitch": fx.BOX_PITCH, "centers": [list(c) for c in fx.BOX_CENTERS]},
    "required_cues": ["T", "F", "A"],
    "protected_rects": [list(r) for r in fx.PROTECTED],
    "ignore_rects": [list(r) for r in fx.IGNORE],
    "note": "SYNTHETIC_TEST レイアウトの再生成（template-build のテスト用）",
}


def blank_norm(kind="back") -> np.ndarray:
    """1011x638 に正規化済みの空欄カード（rectify の出力に相当）。"""
    return cv2.resize(fx.draw_card(kind, blank=True), (fx.NW, fx.NH), interpolation=cv2.INTER_AREA)


def write_samples(d, n=3, kind="back"):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        ok, buf = cv2.imencode(".png", blank_norm(kind))
        assert ok
        (d / f"sample_{i}.png").write_bytes(buf.tobytes())
    return d


# ---------------------------------------------------------------- build_template
def test_fewer_than_min_samples_is_error(cfg, tmp_path):
    """設計書 5.4 C1「様式ごとに5枚以上」は緩めない（既定はエラー）。"""
    d = write_samples(tmp_path / "in", 3)
    with pytest.raises(ConfigError, match="allow-fewer"):
        build.build_template(sorted(d.iterdir()), tmp_path / "t" / "x_v1", cfg, template_id="x_v1", spec=SPEC)


def test_allow_fewer_writes_provisional_template(cfg, tmp_path):
    d = write_samples(tmp_path / "in", 3)
    out = tmp_path / "t" / "prov_v1"
    meta = build.build_template(sorted(d.iterdir()), out, cfg, template_id="prov_v1", spec=SPEC, allow_fewer=True)
    assert meta["min_samples_met"] is False and meta["n_samples"] == 3 and meta["min_samples_required"] == 5
    # 書いたのは template.json と罫線マップ・印刷物マップだけ（サンプル画像を複製しない）
    assert sorted(p.name for p in out.iterdir()) == ["print.png", "ruling.png", "template.json"]
    tmpls, _ = load_templates(out.parent, cfg)
    assert [t.id for t in tmpls] == ["prov_v1"]


def test_enough_samples_sets_min_samples_met(cfg, tmp_path):
    d = write_samples(tmp_path / "in", 5)
    meta = build.build_template(sorted(d.iterdir()), tmp_path / "t" / "ok_v1", cfg, template_id="ok_v1", spec=SPEC)
    assert meta["min_samples_met"] is True and meta["align_correlations"] == [1.0] * 5


def test_built_ruling_matches_committed_synthetic_template(cfg, tmp_path):
    """同一レイアウトの空欄サンプルから作ると、同梱の合成テンプレートと同じ罫線マップになる。"""
    d = write_samples(tmp_path / "in", 3)
    out = tmp_path / "t" / "synthetic_back_v1"
    build.build_template(sorted(d.iterdir()), out, cfg, template_id="synthetic_back_v1", spec=SPEC, allow_fewer=True)
    for name in ("ruling.png", "print.png"):
        assert (out / name).read_bytes() == (ROOT / "templates" / "synthetic_back_v1" / name).read_bytes(), name


def test_expected_mask_area_defaults_from_field_rect(cfg, tmp_path):
    d = write_samples(tmp_path / "in", 3)
    meta = build.build_template(sorted(d.iterdir()), tmp_path / "t" / "a_v1", cfg, template_id="a_v1",
                                spec=SPEC, allow_fewer=True)
    R = tuple(SPEC["field_rect"])
    assert meta["expected_mask_area"] == pytest.approx(round(rect_area(padded(R, base_pads(R, cfg))), 2))


def test_sample_of_wrong_size_is_error(cfg, tmp_path):
    d = write_samples(tmp_path / "in", 3)
    ok, buf = cv2.imencode(".png", cv2.resize(blank_norm(), (800, 505)))
    assert ok
    (d / "small.png").write_bytes(buf.tobytes())
    with pytest.raises(ConfigError, match="rectify"):
        build.build_template(sorted(d.iterdir()), tmp_path / "t" / "x_v1", cfg, template_id="x_v1",
                             spec=SPEC, allow_fewer=True)


def test_mixed_style_sample_is_error(cfg, tmp_path):
    """別様式（表面相当）を混ぜると ECC 相関が足りずエラー。違う様式を平均で潰さない。"""
    d = write_samples(tmp_path / "in", 3)
    ok, buf = cv2.imencode(".png", blank_norm("front"))
    assert ok
    (d / "front.png").write_bytes(buf.tobytes())
    with pytest.raises(ConfigError, match="位置合わせ"):
        build.build_template(sorted(d.iterdir()), tmp_path / "t" / "x_v1", cfg, template_id="x_v1",
                             spec=SPEC, allow_fewer=True)


@pytest.mark.parametrize("drop", ["field_rect", "required_cues", "protected_rects", "ignore_rects", "anchor_type"])
def test_spec_missing_measurement_is_error(cfg, tmp_path, drop):
    """座標は測定値。無いまま推測で埋めない（推測で埋めると別の場所を塗る経路になる）。"""
    d = write_samples(tmp_path / "in", 3)
    spec = {k: v for k, v in SPEC.items() if k != drop}
    with pytest.raises(ConfigError, match="測定値"):
        build.build_template(sorted(d.iterdir()), tmp_path / "t" / "x_v1", cfg, template_id="x_v1",
                             spec=spec, allow_fewer=True)


def test_anchor_none_without_f_or_a_cue_is_rejected(cfg, tmp_path):
    """anchor_type=none でも「T だけ」は不可（親レビュー P2-2 の方針）。"""
    d = write_samples(tmp_path / "in", 3)
    spec = dict(SPEC, anchor_type="none", anchor=None, required_cues=["T"])
    with pytest.raises(ConfigError, match="T だけ"):
        build.build_template(sorted(d.iterdir()), tmp_path / "t" / "x_v1", cfg, template_id="x_v1",
                             spec=spec, allow_fewer=True)
    assert not (tmp_path / "t" / "x_v1").exists()  # 落ちたテンプレートを残さない


def test_cli_template_build_end_to_end(cfg, tmp_path, capsys):
    d = write_samples(tmp_path / "in", 3)
    specp = tmp_path / "spec.json"
    specp.write_text(json.dumps(SPEC, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "t" / "rebuilt_v1"
    assert cli.main(["template-build", "--in", str(d), "--out", str(out), "--id", "rebuilt_v1",
                     "--spec", str(specp), "--allow-fewer"]) == 0
    tmpls, _ = load_templates(out.parent, cfg)
    t = next(t for t in tmpls if t.id == "rebuilt_v1")
    assert t.positive and t.meta["min_samples_met"] is False
    # 作ったテンプレートで 1 件処理できる（罫線マップが同梱テンプレートと同等なので ok になる）
    rec, _, _ = run_fixture(tmp_path, fx.make({"seed": 1}), cfg, (tmpls, "x"))
    assert rec["status"] == "ok"


def test_cli_template_build_refuses_too_few_without_flag(tmp_path, capsys):
    d = write_samples(tmp_path / "in", 2)
    specp = tmp_path / "spec.json"
    specp.write_text(json.dumps(SPEC), encoding="utf-8")
    rc = cli.main(["template-build", "--in", str(d), "--out", str(tmp_path / "t" / "x_v1"),
                   "--id", "x_v1", "--spec", str(specp)])
    assert rc == 1 and "--allow-fewer" in capsys.readouterr().err
    assert not (tmp_path / "t" / "x_v1").exists()


# ---------------------------------------------------------------- C_FRAME_FILLED
def as_fixture(img) -> fx.Fixture:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return fx.Fixture(img, buf.tobytes(), ".png", None, None)


def card_crop(f, size=(fx.NW, fx.NH)):
    """トリミング済みカード（余白ゼロで切り出した写真）の合成版。角丸の外側に背景が少し写り、JPEG の劣化も入る。"""
    W, H = size
    M = cv2.getPerspectiveTransform(f.quad.astype(np.float32), np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(f.img, M, (W, H), flags=cv2.INTER_LINEAR)


@pytest.fixture(scope="module")
def frame_filled_fixture():
    """カードが画像いっぱい（余白 0）。画像全体の輪郭だけが唯一の四角形になる。"""
    return as_fixture(blank_norm())


def test_frame_filled_card_is_rejected_with_code(cfg, tmpls, tmp_path, frame_filled_fixture):
    res = find_candidates(frame_filled_fixture.img, cfg, tmpls[0])
    assert res["candidates"] == [] and res["rejected"].get("C_FRAME_FILLED", 0) >= 1
    assert "C_AREA" not in res["rejected"]  # 「面積が大きすぎる」ではなく専用コードで落とす
    ff = res["frame_filled"]
    assert ff["status"] == "filled"
    assert ff["zncc"] >= cfg.card.frame_filled_zncc_min
    assert ff["zncc"] > ff["inset_max"]
    rec, out, _ = run_fixture(tmp_path, frame_filled_fixture, cfg, tmpls)
    assert rec["status"] == "review"
    assert "C_FRAME_FILLED" in reason_codes(rec)
    assert rec["primary_reason"] == "C_FRAME_FILLED"
    assert not (out / "ok" / f"{rec['job_id']}__masked.png").exists()


def test_square_frame_is_not_frame_filled(cfg, tmpls):
    """アスペクトが 1.585±5% から外れる全面の四角形は C_FRAME_FILLED にしない（正方形の写真など）。"""
    img = np.full((1200, 1200, 3), 235, np.uint8)
    cv2.rectangle(img, (0, 0), (1199, 1199), (30, 30, 30), 3)
    res = find_candidates(img, cfg, tmpls[0])
    assert res["candidates"] == []
    assert "C_FRAME_FILLED" not in res["rejected"] and res["rejected"].get("C_AREA", 0) >= 1


def test_normal_card_is_not_frame_filled(cfg, tmpls):
    """通常の余白のある写真は C_FRAME_FILLED にならない（誤爆しない）。"""
    res = find_candidates(fx.make({"seed": 1}).img, cfg, tmpls[0])
    assert "C_FRAME_FILLED" not in res["rejected"] and len(res["candidates"]) >= 1
    assert res["frame_filled"]["status"] in ("no_outline", "has_candidates")


def test_frame_match_tie_is_not_filled(cfg, tmpls, monkeypatch):
    """縮めた枠の照合が全体の枠と同じ値（同点）なら filled にしない。"""
    from donor_mask import orient

    img = blank_norm()
    base = orient.frame_match(img, tmpls[0], cfg)
    assert base["filled"] is True  # 前提: 同点でなければ filled
    monkeypatch.setattr(orient, "score_quad", lambda *a, **k: base["zncc"])
    m = orient.frame_match(img, tmpls[0], cfg)
    assert m["zncc"] == m["inset_max"] and m["filled"] is False


def test_pipeline_keeps_frame_filled_out_of_ok():
    """C_FRAME_FILLED は ok 経路を開かない（条件付きフォールバックを入れていない）。"""
    import inspect

    src = inspect.getsource(card.find_candidates)
    assert "C_FRAME_FILLED" in src and "is_frame_filled" in inspect.getsource(card)
    assert "is_frame_filled" in src
    # pipeline 側に「枠いっぱいなら全体をカードとみなす」経路が無いこと。判定は card にしか無い
    psrc = inspect.getsource(pipeline.process_file)
    assert "FRAME_FILLED" not in psrc and "frame_filled" not in psrc
    allsrc = inspect.getsource(pipeline)
    assert "is_frame_filled" not in allsrc and "frame_match" not in allsrc


@pytest.mark.parametrize("wh", [(1104, 700), (1200, 757)])
def test_unrelated_card_aspect_photo_is_not_frame_filled(cfg, tmpls, tmp_path, wh):
    """書類もカードも無い、カード比に近い写真（机・カップ・鉛筆）。アスペクトだけで C_FRAME_FILLED にしない。"""
    W, H = wh
    bg = fx.background("plain", W, H, 0)
    cv2.ellipse(bg, (int(W * .3), int(H * .4)), (int(W * .12), int(H * .15)), 0, 0, 360, (60, 60, 140), -1,
                cv2.LINE_AA)
    cv2.line(bg, (int(W * .55), int(H * .2)), (int(W * .9), int(H * .75)), (30, 30, 30), 6, cv2.LINE_AA)
    res = find_candidates(bg, cfg, tmpls[0])
    assert res["candidates"] == []
    assert "C_FRAME_FILLED" not in res["rejected"]
    assert res["frame_filled"]["status"] == "not_filled"
    assert res["frame_filled"]["zncc"] < cfg.card.frame_filled_zncc_min
    rec, _out, _ = run_fixture(tmp_path, as_fixture(bg), cfg, tmpls)
    assert rec["status"] == "review"
    assert reason_codes(rec) == ["C_NO_CARD"]
    assert rec["primary_reason"] == "C_NO_CARD"


@pytest.mark.parametrize("case,size", [({"seed": 1, "angle": 3}, (1011, 638)),
                                       ({"seed": 2, "angle": -5, "persp": 0.03}, (1200, 757))])
def test_trimmed_card_crop_is_frame_filled(cfg, tmpls, tmp_path, case, size):
    """真陽性を落とさない: 余白ゼロでトリミングしたカードは C_FRAME_FILLED（再撮影を案内）。"""
    img = card_crop(fx.make(case), size)
    res = find_candidates(img, cfg, tmpls[0])
    assert res["candidates"] == []
    assert res["rejected"]["C_FRAME_FILLED"] == 1
    ff = res["frame_filled"]
    assert ff["status"] == "filled" and ff["template"] == "synthetic_back_v1"
    assert ff["rot"] == 0 and ff["mirror"] is False
    rec, out, _ = run_fixture(tmp_path, as_fixture(img), cfg, tmpls)
    assert "C_FRAME_FILLED" in reason_codes(rec)
    assert rec["primary_reason"] == "C_FRAME_FILLED"
    assert not (out / "ok").exists() or list((out / "ok").glob("*")) == []


def test_small_card_in_card_aspect_photo_reports_its_own_reason(cfg, tmpls, tmp_path):
    """カード比の写真に小さなカードが写っている（面積比 < 0.15）。外枠は画像の縁で、理由は C_AREA。"""
    f = fx.make({"seed": 11, "scale": 0.40})
    res = find_candidates(f.img, cfg, tmpls[0])
    assert "C_FRAME_FILLED" not in res["rejected"]
    assert res["rejected"].get("C_AREA", 0) >= 1
    assert res["frame_filled"]["status"] == "not_filled"
    rec, _out, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert reason_codes(rec) == ["C_NO_CARD", "C_AREA"]
    assert rec["primary_reason"] == "C_AREA"


def test_inset_condition_rejects_card_with_small_margin(cfg, tmpls):
    """ZNCC だけなら通る（余白 3% のカードでも画像全体が罫線に 0.58 で合う）が、縮めた枠のほうが高いので filled にしない。"""
    m = card.is_frame_filled(fx.make({"seed": 1}).img, tmpls[0], cfg)
    assert m["zncc"] >= cfg.card.frame_filled_zncc_min
    assert m["inset_max"] > m["zncc"]
    assert m["status"] == "not_filled"


def test_find_candidates_requires_explicit_templates(cfg):
    """templates は必須引数。None は「照合しない」＝C_FRAME_UNVERIFIED（黙って旧挙動や無判定にしない）。"""
    with pytest.raises(TypeError):
        find_candidates(blank_norm(), cfg)
    res = find_candidates(blank_norm(), cfg, None)
    assert res["rejected"].get("C_FRAME_UNVERIFIED") == 1
    assert "C_FRAME_FILLED" not in res["rejected"]
    assert res["frame_filled"]["status"] == "no_templates"


def test_primary_reason_order():
    assert card.primary_rejection({"C_ASPECT": 6, "C_FRAME_FILLED": 1}) == "C_FRAME_FILLED"
    assert card.primary_rejection({"C_ANGLE": 3, "C_NOT_QUAD": 2}) == "C_ANGLE"
    assert card.primary_rejection({"C_TOUCHES_BORDER": 1, "C_FRAME_FILLED": 1, "C_ASPECT": 10}) == "C_TOUCHES_BORDER"
    assert card.primary_rejection({}) is None
    assert card.primary_rejection({"C_NOT_QUAD": 2}) is None
    assert pipeline.primary_reason([]) is None
    rs = [{"code": "Q_BLUR", "stage": "S5", "metric": "x", "value": 1, "threshold": 2},
          {"code": "F_FRAME_NOT_FOUND", "stage": "S6", "metric": "y", "value": 0, "threshold": 1}]
    assert pipeline.primary_reason(rs) == "Q_BLUR"
    no_card = [{"code": "C_NO_CARD", "stage": "S1", "metric": "candidates", "value": 0, "threshold": 1}]
    assert pipeline.primary_reason(no_card) == "C_NO_CARD"
    no_card += [{"code": c, "stage": "S1", "metric": "rejected_candidates", "value": n, "threshold": 0}
                for c, n in (("C_ASPECT", 6), ("C_FRAME_FILLED", 1))]
    assert pipeline.primary_reason(no_card) == "C_FRAME_FILLED"


def test_ok_record_has_null_primary_reason(cfg, tmpls, tmp_path):
    rec, _out, _ = run_fixture(tmp_path, fx.make({"seed": 1}), cfg, tmpls)
    assert rec["status"] == "ok" and rec["primary_reason"] is None
    keys = list(rec)
    assert keys.index("primary_reason") == keys.index("reasons") + 1


def test_diagnostic_s1_log_matches_run(cfg, tmpls, tmp_path, frame_filled_fixture):
    """run と diagnostic の S1 ログは同じ（frame_filled を含む）。"""
    rec, _out, p = run_fixture(tmp_path, frame_filled_fixture, cfg, tmpls)
    drec, _ = pipeline.diagnose_file(p, tmp_path / "diag", cfg, *tmpls)
    assert drec["stages"]["S1"] == rec["stages"]["S1"]
    assert drec["stages"]["S1"]["frame_filled"]["status"] == "filled"
    assert drec["primary_reason"] == "C_FRAME_FILLED"


_SC = "frame_filled_inset_scales = [0.97, 0.94, 0.91, 0.88]"


@pytest.mark.parametrize("old,new", [
    ("frame_filled_zncc_min = 0.40", "frame_filled_zncc_min = 0"),
    ("frame_filled_zncc_min = 0.40", "frame_filled_zncc_min = 1.0"),
    (_SC, "frame_filled_inset_scales = []"),
    (_SC, "frame_filled_inset_scales = [1.0]"),
    (_SC, "frame_filled_inset_scales = [0.4]"),
    (_SC, "frame_filled_inset_scales = [0.88, 0.97]"),
    (_SC, "frame_filled_inset_scales = [0.97, 0.97]"),
])
def test_frame_gate_config_is_validated(tmp_path, old, new):
    src = (ROOT / "config.toml").read_text(encoding="utf-8")
    assert old in src
    p = tmp_path / "bad.toml"
    p.write_text(src.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


F_RULES_SPEC = {
    "h_lines": [{"y": [70.0, 80.0], "x": [60.0, 900.0]},
                {"y": [300.0, 310.0], "x": [300.0, 700.0]},
                {"y": [400.0, 410.0], "x": [300.0, 700.0]}],
    "v_lines": [{"x": [40.0, 60.0], "y": [300.0, 400.0]},
                {"x": [900.0, 920.0], "y": [60.0, 200.0]}],
}


def test_spec_f_rules_goes_through_to_template_json(cfg, tmp_path):
    """--spec の f_rules は build の許可キーに入っていて、そのまま template.json に書かれ、ローダーが読める。"""
    assert "f_rules" in build.META_KEYS
    d = write_samples(tmp_path / "in", 3)
    specp = tmp_path / "spec.json"
    specp.write_text(json.dumps(dict(SPEC, f_rules=F_RULES_SPEC), ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "t" / "frules_v1"
    assert cli.main(["template-build", "--in", str(d), "--out", str(out), "--id", "frules_v1",
                     "--spec", str(specp), "--allow-fewer"]) == 0
    written = json.loads((out / "template.json").read_text(encoding="utf-8"))
    assert written["f_rules"] == F_RULES_SPEC
    tmpls, _ = load_templates(out.parent, cfg)
    assert next(t for t in tmpls if t.id == "frules_v1").meta["f_rules"] == F_RULES_SPEC


def test_spec_without_f_rules_writes_null(cfg, tmp_path):
    d = write_samples(tmp_path / "in", 3)
    meta = build.build_template(sorted(d.iterdir()), tmp_path / "t" / "nof_v1", cfg, template_id="nof_v1",
                                spec=SPEC, allow_fewer=True)
    assert meta["f_rules"] is None
