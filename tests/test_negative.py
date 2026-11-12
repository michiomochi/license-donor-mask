"""陰性・失敗注入（合成）: 静かに成功しない（review になり、ok/ に何も出ない）ことを固定する。"""
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import ROOT, assert_ok_invariants, ok_files, reason_codes, run_fixture

import fixtures_gen as fx
from donor_mask import imgio, pipeline, verify
from donor_mask.config import ConfigError, load_config


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
    best, second, mirror, rs = pipeline.orient.select_unique([h(0, 0.9), h(180, 0.9 - 1e-9)], cands, cfg)
    assert "O_AMBIGUOUS" in [r["code"] for r in rs]
    best, second, mirror, rs = pipeline.orient.select_unique([h(180, 0.9), h(0, 0.9)], cands, cfg)
    assert "O_AMBIGUOUS" in [r["code"] for r in rs]


# ---- §1.5c 鏡像双子（twin）: twin だけを主マージンから外し、M2 で鏡像を別に見る
def _qcands():
    return [{"quad": np.array([[0, 0], [100, 0], [100, 60], [0, 60]], float)},
            {"quad": np.array([[200, 0], [300, 0], [300, 60], [200, 60]], float)}]


def _qh(rot, z, mirror=False, cand=0, template="t"):
    return {"cand": cand, "rot": rot, "mirror": mirror, "template": template,
            "kind": "positive", "label": "back", "zncc": z}


def _codes(rs):
    return [r["code"] for r in rs]


def test_twin_is_excluded_from_the_main_margin(cfg):
    """twin（同候補・同rot・mirror 反転）は second に入らない。3位との差で O_AMBIGUOUS を判定する。"""
    best, second, mirror, rs = pipeline.orient.select_unique(
        [_qh(0, 0.90), _qh(0, 0.70, mirror=True), _qh(180, 0.30)], _qcands(), cfg)
    assert best["zncc"] == pytest.approx(0.90) and best["mirror"] is False
    assert second["rot"] == 180, "twin が second に残っている"
    assert mirror["twin"]["mirror"] is True
    assert mirror["d_mirror"] == pytest.approx(0.20)
    assert "O_AMBIGUOUS" not in _codes(rs)


def test_mirror_margin_shortfall_is_o_mirror_ambiguous(cfg):
    """M2: best − twin < mirror_margin なら O_MIRROR_AMBIGUOUS（twin を外しても通さない）。"""
    m = cfg.select.mirror_margin
    best, second, mirror, rs = pipeline.orient.select_unique(
        [_qh(0, 0.90), _qh(0, 0.90 - m / 2, mirror=True), _qh(180, 0.30)], _qcands(), cfg)
    assert mirror["d_mirror"] < m
    assert second["rot"] == 180, "twin が second に残っている"
    assert "O_MIRROR_AMBIGUOUS" in _codes(rs)
    assert "O_AMBIGUOUS" not in _codes(rs)


def test_upside_down_is_not_a_twin_and_still_competes(cfg):
    """上下反転（鏡像＋180°）は twin ではない。主マージンの比較対象に残る。"""
    best, second, mirror, rs = pipeline.orient.select_unique(
        [_qh(0, 0.90), _qh(180, 0.88, mirror=True)], _qcands(), cfg)
    assert mirror["twin"] is None and mirror["d_mirror"] is None
    assert second["rot"] == 180
    assert "O_AMBIGUOUS" in _codes(rs)


def test_twin_requires_same_candidate_same_rotation_and_same_family(cfg):
    """別候補・別rot・別ファミリー（field_rect が違うテンプレート）は twin にしない。"""
    for hyps in ([_qh(0, 0.90), _qh(0, 0.88, mirror=True, cand=1)],
                 [_qh(0, 0.90), _qh(90, 0.88, mirror=True)],
                 [_qh(0, 0.90), _qh(0, 0.88, mirror=True, template="u")]):
        best, second, mirror, rs = pipeline.orient.select_unique(hyps, _qcands(), cfg)
        assert mirror["twin"] is None
        assert second is not None and second["zncc"] == pytest.approx(0.88)
        assert "O_AMBIGUOUS" in _codes(rs)


def test_mirror_gate_cannot_be_disabled_in_config(tmp_path):
    """§1.5c の不変条件: twin 除外と M2・M3 は一体。緩める設定は起動しない（ConfigError）。"""
    src = (ROOT / "config.toml").read_text()
    for old, new in (("mirror_margin = 0.03", "mirror_margin = 0.0"),
                     ("mirror_m3 = \"unimplemented\"", "mirror_m3 = \"off\""),
                     ("family_iou = 0.90", "family_iou = 0.0"),
                     ("mirror_margin = 0.03", "mirror_margin = -0.03")):
        p = tmp_path / "bad.toml"
        p.write_text(src.replace(old, new))
        with pytest.raises(ConfigError):
            load_config(p)


def test_m3_unimplemented_is_logged_not_silently_dropped(cfg, tmpls, tmp_path):
    """M3（印刷数字の左右反転照合）は v2.1 の一部未実装。黙って無効化せず、保留をログに残す。"""
    rec, _, _ = run_fixture(tmp_path, fx.make({"seed": 2}), cfg, tmpls)
    m3 = rec["stages"]["S3"]["mirror"]["m3"]
    assert m3["status"] == "unimplemented" and m3["pending_code"] == "F_ANCHOR_MIRRORED"


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


# ---- S9 の fail-closed（設計 6.3）: review なのに ok/ に masked が残ると下流が誤納品する
def _staging_files(out):
    d = out / "ok" / ".staging"
    return sorted(f.name for f in d.iterdir()) if d.exists() else []


def test_preview_render_failure_leaves_ok_empty(cfg, tmpls, tmp_path, monkeypatch):
    """プレビュー生成が落ちたら review。先に確定した masked を ok/ に残さない。"""
    def boom(*a, **k):
        raise RuntimeError("injected preview failure")
    monkeypatch.setattr(pipeline, "draw_preview", boom)
    f = fx.make({"seed": 1})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert_review(rec, od, ["X_EXCEPTION"])
    assert _staging_files(od) == []


def test_original_changed_after_verify_leaves_ok_empty(cfg, tmpls, tmp_path, monkeypatch):
    """V6 通過後・確定前に原本が変わったら ok/ に何も残さず review + halt。

    この時点で ok/ が空であること（＝確定より前に確認している）も固定する。
    """
    templates, tsha = tmpls
    f = fx.make({"seed": 1})
    d = tmp_path / "in"
    d.mkdir()
    p = d / ("SYNTHETIC_TEST_changed" + f.ext)
    p.write_bytes(f.data)
    out = tmp_path / "out"
    real = verify.v6_original
    state = {"bumped": False}

    def bump(path, before):
        r = real(path, before)  # まずは実物で判定する（以降は正直な判定に戻す）
        if not r["pass"]:  # 判定が落ちた時点でも ok/ は空（＝確定前に確認している）
            assert ok_files(out) == []
        if not state["bumped"] and r["pass"]:
            assert ok_files(out) == []  # mtime を進める時点でまだ確定していない
            state["bumped"] = True
            st = os.stat(path)
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        return r

    monkeypatch.setattr(verify, "v6_original", bump)
    rec, halt = pipeline.process_file(p, out, cfg, templates, tsha)
    assert halt is True
    assert rec["status"] == "review"
    assert "X_ORIGINAL_CHANGED" in reason_codes(rec)
    assert ok_files(out) == []
    assert rec["outputs"]["masked"] is None
    assert rec["outputs"]["preview"] is None


def test_ok_case_commits_masked_and_preview_together(cfg, tmpls, tmp_path):
    """正常系: ok/ には masked と preview の2つだけが揃い、staging は空。"""
    f = fx.make({"seed": 1})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    assert rec["status"] == "ok", rec["reasons"]
    job = rec["job_id"]
    assert ok_files(od) == [f"{job}__masked.png", f"{job}__preview.jpg"]
    assert _staging_files(od) == []


def test_replace_failure_after_masked_placed_rolls_back(cfg, tmpls, tmp_path, monkeypatch):
    """masked を ok/ へ置いた直後の確定失敗でも ok/ を空にして review に落とす。"""
    real = pipeline.os.replace
    seen: dict = {"ok_at_failure": None}

    def flaky(src, dst, *a, **k):
        # 置き場所に依存させる（通算回数に依存すると、無関係な実装変更で注入点がずれる）
        if str(dst).endswith("__preview.jpg"):
            seen["ok_at_failure"] = sorted(p.name for p in Path(dst).parent.iterdir() if p.is_file())
            raise OSError("injected replace failure")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(pipeline.os, "replace", flaky)
    f = fx.make({"seed": 1})
    rec, od, _ = run_fixture(tmp_path, f, cfg, tmpls)
    # 契約 → 内部の順で確認する
    assert rec["status"] == "review"
    assert ok_files(od) == []
    assert seen["ok_at_failure"] == [f"{rec['job_id']}__masked.png"]  # 失敗時点で masked は ok/ にあった（rollback を実際に検証）
    assert rec["outputs"]["masked"] is None
    assert _staging_files(od) == []


def test_original_changed_after_commit_rolls_back_ok(cfg, tmpls, tmp_path, monkeypatch):
    """S9 通過後（ok 判定後）の競合を末尾の再確認で拾い、ok/ を巻き戻す。"""
    templates, tsha = tmpls
    f = fx.make({"seed": 1})
    d = tmp_path / "in"
    d.mkdir()
    p = d / ("SYNTHETIC_TEST_late_change" + f.ext)
    p.write_bytes(f.data)
    out = tmp_path / "out"
    real_replace = pipeline.os.replace

    def bump_after_commit(src, dst, *a, **k):
        r = real_replace(src, dst, *a, **k)
        if str(dst).endswith("__preview.jpg"):  # masked と preview が揃って ok/ に確定した直後に原本が変わる競合
            st = os.stat(p)
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        return r

    monkeypatch.setattr(pipeline.os, "replace", bump_after_commit)
    rec, halt = pipeline.process_file(p, out, cfg, templates, tsha)
    assert halt is True
    assert rec["status"] == "review"
    assert "X_ORIGINAL_CHANGED" in reason_codes(rec)
    assert ok_files(out) == []
    assert rec["outputs"]["masked"] is None


def _run_one(tmp_path, fixture, cfg, tmpls, name):
    """halt=True が想定のケース用: run_fixture は halt を許さないので直接呼ぶ。"""
    templates, tsha = tmpls
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    p = d / (name + fixture.ext)
    p.write_bytes(fixture.data)
    out = tmp_path / "out"
    rec, halt = pipeline.process_file(p, out, cfg, templates, tsha)
    return rec, out, halt


def test_staging_cleanup_failure_keeps_the_real_reason(cfg, tmpls, tmp_path, monkeypatch):
    """staging の掃除が失敗しても真因の理由コードを X_EXCEPTION に置換せず、もう片方の掃除も続ける。"""
    real_unlink = Path.unlink

    def flaky_unlink(self, *a, **k):
        if self.name.endswith("__masked.png") and self.parent.name == ".staging":
            raise OSError("injected unlink failure")
        return real_unlink(self, *a, **k)

    real_read = pipeline.imgio.read_array
    state = {"in_s9": False}

    def mark(p):  # staging の読み戻しが走ったら S9 の検証フェーズに入ったと分かる
        arr = real_read(p)
        if str(p).endswith("__masked.png"):
            state["in_s9"] = True
        return arr

    real_v6 = verify.v6_original

    def v6_fail_in_s9(path, before):
        r = real_v6(path, before)
        # staging の読み戻し後（＝S9 の検証フェーズ）は常に False。in_s9 はラッチなので一度立つと以後の V6
        # （末尾の再確認を含む）も False を返す。ここでの目的は V6 失敗で StageFail を in-flight にすること。
        if state["in_s9"]:
            return {"pass": False, "code": "X_ORIGINAL_CHANGED", "metrics": r["metrics"]}
        return r

    monkeypatch.setattr(pipeline.imgio, "read_array", mark)
    monkeypatch.setattr(verify, "v6_original", v6_fail_in_s9)
    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    f = fx.make({"seed": 1})
    rec, od, _ = _run_one(tmp_path, f, cfg, tmpls, "SYNTHETIC_TEST_cleanup")
    assert rec["status"] == "review"
    codes = reason_codes(rec)
    assert "X_ORIGINAL_CHANGED" in codes, codes
    assert "X_EXCEPTION" not in codes, codes  # 掃除の失敗で真因が潰れていない
    assert ok_files(od) == []
    # 2つ目（preview）の掃除まで走る。消せなかった masked だけが staging に残る
    assert _staging_files(od) == [f"{rec['job_id']}__masked.png"]
    cleanup = rec["stages"].get("S9_cleanup", {})
    assert list(cleanup) == [f"{rec['job_id']}__masked.png"]
    assert "OSError" in list(cleanup.values())[0]


def test_ok_cleanup_failure_keeps_the_path_for_visibility(cfg, tmpls, tmp_path, monkeypatch):
    """ok/ の削除に失敗したら、そのパスはログに残す（null にすると残骸に気づけない）。"""
    real_unlink = Path.unlink

    def flaky_unlink(self, *a, **k):
        if self.name.endswith("__masked.png") and self.parent.name == "ok":
            raise OSError("injected unlink failure")
        return real_unlink(self, *a, **k)

    real_replace = pipeline.os.replace

    def bump_after_commit(src, dst, *a, **k):  # 確定後に原本が変わる競合（outputs が埋まった後の巻き戻し）
        r = real_replace(src, dst, *a, **k)
        if str(dst).endswith("__preview.jpg"):
            st = os.stat(p)
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        return r

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    monkeypatch.setattr(pipeline.os, "replace", bump_after_commit)
    f = fx.make({"seed": 1})
    p = tmp_path / "in" / ("SYNTHETIC_TEST_okcleanup" + f.ext)  # _run_one が同じ名前で書き出す
    rec, od, halt = _run_one(tmp_path, f, cfg, tmpls, "SYNTHETIC_TEST_okcleanup")
    assert halt is True
    assert rec["status"] == "error"
    assert "X_OK_CLEANUP_FAILED" in reason_codes(rec)
    assert rec["outputs"]["masked"] is not None  # 残骸のありかをログで示す
    assert rec["outputs"]["masked"].endswith("__masked.png")
    assert Path(rec["outputs"]["masked"]).exists()
    assert rec["outputs"]["preview"] is None  # 消せた方は null


# ---- Issue #4: 解像度ゲートは入力画像の長辺ではなく、S1 後のカード長辺で見る
def _downscaled(long_edge, seed=3):
    import cv2
    f = fx.make({"seed": seed})
    img, _ = imgio.decode(f.data, 10**9)
    s = long_edge / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return small, cv2.imencode(".png", small)[1].tobytes()


@pytest.mark.parametrize("long_edge", [700, 500])
def test_low_res_image_with_large_enough_card_passes_resolution_gate(cfg, tmpls, tmp_path, long_edge):
    """入力画像の長辺が 800px 未満でも、カード長辺が下限以上なら Q_LOW_RES にならない。"""
    small, data = _downscaled(long_edge)
    assert max(small.shape[:2]) < 800
    t, ts = tmpls
    d = tmp_path / "in"
    d.mkdir()
    p = d / "SYNTHETIC_TEST_a.png"
    p.write_bytes(data)
    rec, _ = pipeline.process_file(p, tmp_path / "out", cfg, t, ts)
    assert "Q_LOW_RES" not in reason_codes(rec)
    assert rec["stages"]["S5"]["card_long_edge_px"] >= cfg.input.min_card_long_edge_px


def test_small_card_is_q_low_res_with_card_metric(cfg, tmpls, tmp_path):
    """カードが小さい入力は Q_LOW_RES（metric = card_long_edge_px）で review。ok/ には何も出ない。"""
    _, data = _downscaled(380)
    t, ts = tmpls
    d = tmp_path / "in"
    d.mkdir()
    p = d / "SYNTHETIC_TEST_a.png"
    p.write_bytes(data)
    rec, _ = pipeline.process_file(p, tmp_path / "out", cfg, t, ts)
    low = [r for r in rec["reasons"] if r["code"] == "Q_LOW_RES"]
    assert rec["status"] == "review" and low
    assert low[0]["metric"] == "card_long_edge_px" and low[0]["threshold"] == cfg.input.min_card_long_edge_px
    assert ok_files(tmp_path / "out") == []
