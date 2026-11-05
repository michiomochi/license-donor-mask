"""陰性・失敗注入（合成）: 静かに成功しない（review になり、ok/ に何も出ない）ことを固定する。"""
import numpy as np
import pytest
from conftest import assert_ok_invariants, ok_files, reason_codes, run_fixture

import fixtures_gen as fx
from donor_mask import pipeline, verify


def assert_review(rec, od, codes_any=None):
    assert rec["status"] == "review", (rec["status"], rec["reasons"])
    assert ok_files(od) == []
    assert rec["outputs"]["masked"] is None
    assert rec["reasons"], "理由コードが空"
    if codes_any:
        assert set(codes_any) & set(reason_codes(rec)), (codes_any, reason_codes(rec))


# ---- 試作の失敗2: 極端な斜め（上辺が下辺の 50/60/70%）→ 全体をカードとみなして成功扱い、を再発させない
@pytest.mark.parametrize("ratio", [0.5, 0.6, 0.7])
def test_extreme_perspective_is_review(cfg, tmpls, tmp_path, ratio):
    f = fx.make({"trapezoid": ratio})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert_review(rec, od)
    # 台形のカード自体は S1 で C_PERSPECTIVE として棄却される（内側の構造が別候補として残って S2/S3 で落ちることはある）
    rejected = rec["stages"]["S1"]["rejected"]
    assert rejected.get("C_PERSPECTIVE", 0) >= 1 or "C_PERSPECTIVE" in reason_codes(rec)


# ---- 試作の失敗1: 台紙。紙を選んで被覆率が落ちる、を再発させない（面積最大で選ばない）
@pytest.mark.parametrize("aspect", [1.19, 1.40, 1.585, 1.98])
@pytest.mark.parametrize("card_color", [(196, 222, 200), fx.CARD_COLOR])
def test_paper_backing_never_selects_paper(cfg, tmpls, tmp_path, aspect, card_color):
    f = fx.make({"bg": "paper", "paper_aspect": aspect, "scale": 0.6, "card_color": card_color, "seed": 5})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    if rec["status"] == "ok":
        assert_ok_invariants(f, rec)  # 通るなら包含率 1.000（紙を選んでいたらここで落ちる）
    else:
        assert_review(rec, od)


@pytest.mark.parametrize("aspect", [1.19, 1.40, 1.585, 1.98])
def test_paper_backing_with_detectable_card_picks_the_card_by_template(cfg, tmpls, tmp_path, aspect):
    """カードが台紙と見分けられるコントラストなら、台紙も候補に残り（面積で1つに絞らない）、罫線テンプレート照合で
    カードが選ばれる。包含率 1.000。台紙（紙の四角形）を選んでいたらここで落ちる。"""
    f = fx.make({"bg": "paper", "paper_aspect": aspect, "scale": 0.6, "card_color": (120, 170, 130), "seed": 5})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert rec["status"] == "ok", rec["reasons"]
    assert_ok_invariants(f, rec)
    cands = rec["stages"]["S1"]["candidates"]
    best = rec["stages"]["S3"]["best"]
    chosen = np.array(cands[best["cand"]]["quad"])
    from donor_mask.geom import quad_dist, quad_diag
    assert quad_dist(chosen, f.quad) < 0.01 * quad_diag(f.quad)  # 選ばれたのは紙ではなくカード
    if aspect in (1.40, 1.585):  # 紙がアスペクト許容内 → 2候補が S2 に送られている
        assert len(cands) >= 2
        paper_scores = [h["zncc"] for h in rec["stages"]["S2"]["top"] if h["cand"] != best["cand"] and h["template"] == "synthetic_back_v1"]
        assert all(z < cfg.select.min_zncc for z in paper_scores)


@pytest.mark.parametrize("name,case,codes", [
    ("no_card", {"no_card": True}, ["C_NO_CARD"]),
    ("cut_by_border", {"cut_border": True}, ["C_NO_CARD", "C_TOUCHES_BORDER"]),
    ("two_cards", {"two_cards": True}, ["O_AMBIGUOUS", "C_MULTIPLE_CARDS"]),
    ("front_side", {"kind": "front"}, ["O_FRONT_SIDE"]),
    ("other_card", {"kind": "other"}, ["O_UNSUPPORTED_CARD"]),
    ("mirrored", {"mirror": True}, ["O_MIRRORED"]),
    ("glare_over_field", {"glare": True}, ["Q_GLARE"]),
    ("finger_over_field", {"finger": True}, ["Q_CONTRAST", "F_FRAME_NOT_FOUND", "F_ANCHOR_NOT_FOUND", "F_CUE_DISAGREE", "R_ECC_FAIL"]),
    ("heavy_blur_sigma4", {"blur": 4.0}, ["Q_BLUR", "R_ECC_FAIL", "O_LOW_SCORE", "C_NO_CARD"]),
    ("frame_without_boxes", {"boxes": 0}, ["F_ANCHOR_NOT_FOUND"]),
    ("two_triples", {"extra_triple": True}, ["F_ANCHOR_MULTIPLE"]),
    ("low_resolution", {"scale": 0.4}, ["Q_LOW_RES", "C_NO_CARD"]),
])
def test_negative_cases_go_to_review(cfg, tmpls, tmp_path, name, case, codes):
    f = fx.make(dict(case, seed=11))
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls, name="SYNTHETIC_TEST_" + name)
    assert_review(rec, od, codes)
    assert rec["outputs"]["review"] and rec["outputs"]["review"].endswith("__review.jpg")


def test_upside_down_and_clockwise_are_resolved_not_guessed(cfg, tmpls, tmp_path_factory):
    """向きは8仮説の照合で決まる。上下逆・90/270 度も、一意に決まって ok になるか、決まらなければ review。"""
    for i, a in enumerate((90, 180, 270)):
        f = fx.make({"angle": a, "seed": i})
        rec, od, _ = run_fixture(tmp_path_factory.mktemp(f"rot{a}"), f, cfg, tmpls)
        assert rec["status"] == "ok"
        best = rec["stages"]["S3"]["best"]
        assert best["mirror"] is False
        assert best["zncc"] - rec["stages"]["S3"]["second"]["zncc"] >= cfg.select.min_margin
        assert_ok_invariants(f, rec)


def test_orientation_tie_is_ambiguous_not_order_dependent(cfg):
    """同点（1e-6 以内）は並び順で決めず O_AMBIGUOUS。"""
    cands = [{"quad": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)}]
    h = lambda rot, z: {"cand": 0, "rot": rot, "mirror": False, "template": "t", "kind": "positive", "label": "back", "zncc": z}  # noqa: E731
    best, second, rs = pipeline.orient.select_unique([h(0, 0.9), h(180, 0.9 - 1e-9)], cands, cfg)
    assert "O_AMBIGUOUS" in [r["code"] for r in rs]
    best, second, rs = pipeline.orient.select_unique([h(180, 0.9), h(0, 0.9)], cands, cfg)
    assert "O_AMBIGUOUS" in [r["code"] for r in rs]


# ---- 例外注入: 各段で例外 → X_EXCEPTION で review。ok/ は空のまま
STAGES = {"s1_candidates": "S1", "s2_score": "S2", "s3_select": "S3", "s4_align": "S4", "s5_quality": "S5",
          "s6_locate": "S6", "s7_mask": "S7"}


@pytest.mark.parametrize("stage_fn", sorted(STAGES))
def test_exception_in_any_stage_is_review(cfg, tmpls, tmp_path, monkeypatch, stage_fn):
    def boom(*a, **k):
        raise RuntimeError("injected")
    monkeypatch.setattr(pipeline, stage_fn, boom)
    f = fx.make({"seed": 1})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert_review(rec, od, ["X_EXCEPTION"])
    assert rec["reasons"][0]["stage"] == STAGES[stage_fn]
    assert rec["reasons"][0]["metric"] == "RuntimeError"


def test_exception_in_verify_is_review(cfg, tmpls, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("injected")
    monkeypatch.setattr(verify, "verify_all", boom)
    f = fx.make({"seed": 1})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert_review(rec, od, ["X_EXCEPTION"])


# ---- 位置ずれ注入（試作の「位置がずれても verified=true」の再発防止）
def test_shifted_s6_result_is_caught_by_v1(cfg, tmpls, tmp_path, monkeypatch):
    real = pipeline.s6_locate

    def shifted(aligned, tmpl, cfg_):
        cues, met, rs = real(aligned, tmpl, cfg_)
        d = 30.0  # 正規化座標で 30px（x,y とも）ずらす
        sh = lambda r: None if r is None else (r[0] + d, r[1] + d, r[2] + d, r[3] + d)  # noqa: E731
        for k in ("T", "F", "A", "R"):
            cues[k] = sh(cues[k])
        cues["A_boxes"] = [sh(b) for b in cues["A_boxes"]]
        return cues, met, rs
    monkeypatch.setattr(pipeline, "s6_locate", shifted)
    f = fx.make({"seed": 2})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert_review(rec, od, ["V_INCLUSION"])
    assert rec["stages"]["S8"]["V1"]["metrics"]["contains_field_rect"] is False
    # 一様性だけなら通ってしまう（試作の盲点）ことも確認: V2 は pass、V1 だけが落とした
    assert rec["stages"]["S8"]["V2"]["pass"] is True


def test_shadow_mode_never_writes_ok(cfg, tmpls, tmp_path):
    f = fx.make({"seed": 3})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls, shadow=True)
    assert_review(rec, od, ["X_SHADOW"])
    assert rec["outputs"]["review"]  # 提案をレビュー画像に描く


# ---- 入力の異常
def test_bad_inputs(cfg, tmpls, tmp_path):
    templates, tsha = tmpls
    d = tmp_path / "in"
    d.mkdir()
    out = tmp_path / "out"
    (d / "a.jpg").write_bytes(b"not an image at all")
    (d / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 30)
    (d / "c.jpg").write_bytes(b"")
    small = fx.make({"scale": 1.0})
    import cv2
    tiny = cv2.resize(small.img, (400, 250))
    (d / "d.png").write_bytes(cv2.imencode(".png", tiny)[1].tobytes())
    got = {}
    for f in sorted(d.iterdir()):
        rec, halt = pipeline.process_file(f, out, cfg, templates, tsha)
        got[f.name] = rec
    assert got["a.jpg"]["status"] == "review" and "E_FORMAT_UNSUPPORTED" in reason_codes(got["a.jpg"])
    assert got["b.png"]["status"] in ("review", "error")
    assert got["c.jpg"]["status"] in ("review", "error")
    assert got["d.png"]["status"] == "review" and "Q_LOW_RES" in reason_codes(got["d.png"])
    assert ok_files(out) == []


def test_too_large_image_is_rejected_before_decode(cfg, tmpls, tmp_path):
    import struct
    import zlib
    templates, tsha = tmpls
    w = h = 8000  # 64MP > 50MP。ヘッダだけの PNG（IHDR のみ。デコードに進まないこと）
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
    p = tmp_path / "big.png"
    p.write_bytes(data)
    rec, _ = pipeline.process_file(p, tmp_path / "out", cfg, templates, tsha)
    assert rec["status"] == "review" and "E_TOO_LARGE" in reason_codes(rec)
