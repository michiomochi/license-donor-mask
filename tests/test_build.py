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
from donor_mask import build, cli, pipeline
from donor_mask.card import find_candidates
from donor_mask.config import ConfigError
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
@pytest.fixture(scope="module")
def frame_filled_fixture():
    """カードが画像いっぱい（余白 0）。画像全体の輪郭だけが唯一の四角形になる。"""
    img = blank_norm()
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return fx.Fixture(img, buf.tobytes(), ".png", None, None)


def test_frame_filled_card_is_rejected_with_code(cfg, tmpls, tmp_path, frame_filled_fixture):
    cands = find_candidates(frame_filled_fixture.img, cfg)
    assert cands["candidates"] == [] and cands["rejected"].get("C_FRAME_FILLED", 0) >= 1
    assert "C_AREA" not in cands["rejected"]  # 「面積が大きすぎる」ではなく専用コードで落とす
    rec, out, _ = run_fixture(tmp_path, frame_filled_fixture, cfg, tmpls)
    assert rec["status"] == "review"
    assert "C_FRAME_FILLED" in reason_codes(rec)
    assert not (out / "ok" / f"{rec['job_id']}__masked.png").exists()


def test_square_frame_is_not_frame_filled(cfg):
    """アスペクトが 1.585±5% から外れる全面の四角形は C_FRAME_FILLED にしない（正方形の写真など）。"""
    img = np.full((1200, 1200, 3), 235, np.uint8)
    cv2.rectangle(img, (0, 0), (1199, 1199), (30, 30, 30), 3)
    res = find_candidates(img, cfg)
    assert res["candidates"] == []
    assert "C_FRAME_FILLED" not in res["rejected"] and res["rejected"].get("C_AREA", 0) >= 1


def test_normal_card_is_not_frame_filled(cfg):
    """通常の余白のある写真は C_FRAME_FILLED にならない（誤爆しない）。"""
    res = find_candidates(fx.make({"seed": 1}).img, cfg)
    assert "C_FRAME_FILLED" not in res["rejected"] and len(res["candidates"]) >= 1


def test_pipeline_keeps_frame_filled_out_of_ok():
    """C_FRAME_FILLED は ok 経路を開かない（条件付きフォールバックを入れていない）。"""
    import inspect

    from donor_mask import card
    src = inspect.getsource(card.find_candidates)
    assert "C_FRAME_FILLED" in src and "is_frame_filled" in inspect.getsource(card)
    # pipeline 側に「枠いっぱいなら全体をカードとみなす」経路が無いこと
    psrc = inspect.getsource(pipeline.process_file)
    assert "FRAME_FILLED" not in psrc and "frame_filled" not in psrc
