"""S0〜S9 の統括。各段は「値＋理由コード」を返し、1つでも fail なら review（ok を出さない）。

段の関数（s0_load … s8 の部品）はモジュール属性として呼び出すので、テストで差し替えて例外注入・位置ずれ注入ができる。
全体を包む try/except は例外を握り潰して ok にするのではなく、X_EXCEPTION として review にする。
"""
from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import __version__, card, field, imgio, mask, orient, quality, verify
from .geom import apply_h, edge_lengths, hypothesis_quads, order_quad, polygon_mask, r4, rect_poly

FILL_COLORS = {"black": (0, 0, 0), "white": (255, 255, 255)}  # BGR。単色のみ（noise/ぼかし/モザイクは実装しない）
SCHEMA_VERSION = 1


class StageFail(Exception):
    def __init__(self, stage: str, reasons: list):
        super().__init__(stage)
        self.stage, self.reasons = stage, reasons


def _fail(stage, reasons):
    raise StageFail(stage, [dict(r, stage=stage) for r in reasons])


def _run(stage, fn, *a, **kw):
    """段の呼び出し。想定外の例外は X_EXCEPTION（握り潰して ok にはしない）。"""
    try:
        return fn(*a, **kw)
    except StageFail:
        raise
    except Exception as e:
        raise StageFail(stage, [{"code": "X_EXCEPTION", "stage": stage, "metric": type(e).__name__,
                                 "value": str(e)[:200], "threshold": None}]) from e


# ---------------------------------------------------------------- 段の関数（差し替え可能）
def s0_load(path, cfg):
    return imgio.read_original(Path(path), int(cfg.input.max_pixels))


def s1_candidates(img, cfg, templates):
    return card.find_candidates(img, cfg, templates)


def s1_log(res) -> dict:
    """S1 のログ。process_file と diagnose_file で同一にする（判定そのものは card にしか無い）。"""
    return {"candidates": [{"quad": c["quad"], "maps": c["maps"], "metrics": c["metrics"]} for c in res["candidates"]],
            "rejected": res["rejected"], "frame_filled": res["frame_filled"],
            "color_fallback": res.get("color_fallback", False), "color_rejected": res.get("color_rejected"), "grabcut_fallback": res.get("grabcut_fallback", False), "grabcut_rejected": res.get("grabcut_rejected")}


def primary_reason(reasons: list) -> str | None:
    """reasons から一次理由を1つ選ぶ（reasons の中身・並びは変えない）。

    - 空 → None
    - 先頭が C_NO_CARD（S1 で候補0件）→ S1 の棄却コードを card.REJECT_PRIORITY の順に見て最初のもの。無ければ C_NO_CARD
    - それ以外 → 先頭の理由（失敗した段の最初の理由。diagnostic では最も早い段の理由）
    """
    if not reasons:
        return None
    if reasons[0]["code"] == "C_NO_CARD":
        rejected = {r["code"]: r["value"] for r in reasons
                    if r.get("stage") == "S1" and r.get("metric") == "rejected_candidates"}
        return card.primary_rejection(rejected) or "C_NO_CARD"
    return reasons[0]["code"]


def s2_score(img, cands, templates, cfg):
    return orient.score_hypotheses(img, cands, templates, cfg)


def s3_select(hyps, cands, cfg, templates=None):
    return orient.select_unique(hyps, cands, cfg, templates)


def s4_align(norm, tmpl, cfg):
    return orient.refine_alignment(norm, tmpl, cfg)


def s5_quality(aligned, tmpl, card_long, cfg):
    return quality.check_quality(aligned, tmpl, card_long, cfg)


def s6_locate(aligned, tmpl, cfg):
    return field.locate_field(aligned, tmpl, cfg)


def s7_mask(aligned, tmpl, cues, cfg):
    return mask.build_mask_rect(aligned, tmpl, cues, cfg)


# ---------------------------------------------------------------- ログ
def clean(x, nd=4):
    """JSON 化（numpy 型の変換と桁の固定。同じ入力・設定ならログが完全一致するように）。"""
    if isinstance(x, dict):
        return {str(k): clean(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist(), nd)
    if isinstance(x, (np.floating, float)):
        return round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def sig12(x):
    """幾何（射影行列など）は小数桁ではなく有効数字12桁で固定する。透視項が 1e-7 程度で、桁丸めだと数 px ずれるため。"""
    if isinstance(x, dict):
        return {k: sig12(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sig12(v) for v in x]
    if isinstance(x, np.ndarray):
        return sig12(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(f"{float(x):.12g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def new_rec(path, cfg, tmpl_sha, mode="run"):
    return {
        "schema_version": SCHEMA_VERSION, "mode": mode, "job_id": None,
        "input": {"path": str(Path(path).resolve())},
        "versions": {"tool": __version__, "python": platform.python_version(), "opencv": cv2.__version__,
                     "numpy": np.__version__},
        "config_sha256": cfg.sha256, "templates_sha256": tmpl_sha,
        "status": None, "reasons": [], "primary_reason": None, "stages": {}, "geometry": None,
        "outputs": {"masked": None, "masked_sha256": None, "preview": None, "review": None},
        "original_unchanged": None,
        "nondeterministic": {"started_at": None, "elapsed_ms": None},
    }



def apply_m3(img, cands, best, mirror, rs, templates, cfg):
    """§1.5c M3。mirror_m3 = digits23 のとき、M2（O_MIRROR_AMBIGUOUS）だけが理由で止まる非鏡像の best に限り、
    印刷数字 2・3 の左右反転照合で曖昧さを解く。通ったら O_MIRROR_AMBIGUOUS を外す（他の理由・O_MIRRORED は残す）。
    unimplemented のときは従来どおり判定を保留してログに残すだけ。戻りは (mirror, rs)。
    """
    mirror = dict(mirror or {})
    if str(cfg.select.mirror_m3) == "unimplemented":
        mirror["m3"] = {"status": "unimplemented", "pending_code": "F_ANCHOR_MIRRORED",
                        "note": "印刷数字の左右反転照合（§1.5c M3）は未実装。判定は M2 まで"}
        return mirror, rs
    waivable = {"O_MIRROR_AMBIGUOUS"}
    if int(cfg.select.m3_waives_score) == 1:
        # 数字 2・3 が強く一致する（正しい向きで一意）なら、斜め・低解像度で罫線の ZNCC が低い／別仮説と近いだけの理由は解く。
        # S4〜S7（ECC 相関・罫線 F・数字 A・V1〜V7）は変えない
        waivable |= {"O_LOW_SCORE", "O_AMBIGUOUS"}
    amb = [r for r in rs if r["code"] in waivable]
    tmpl = next((t for t in templates if best is not None and t.id == best["template"]), None)
    if not amb or any(r["code"] not in waivable and r["code"] != "O_MIRRORED" for r in rs) \
            or best is None or best["mirror"] or tmpl is None or not tmpl.positive:
        mirror["m3"] = {"status": "not_needed", "mode": "digits23"}
        return mirror, rs
    quad = hypothesis_quads(cands[best["cand"]]["quad"])[best["rot"] // 90][2]
    norm, _ = orient.normalize(img, quad, cfg)
    m3 = orient.mirror_m3(norm, tmpl, cfg)
    mirror["m3"] = dict(m3, mode="digits23")
    if m3["passed"]:
        rs = [r for r in rs if r["code"] not in waivable]
    return mirror, rs

def _add(rec, stage, value):
    rec["stages"][stage] = clean(value)


def _out_paths(out_dir: Path, job: str):
    return {"masked": out_dir / "ok" / f"{job}__masked.png", "preview": out_dir / "ok" / f"{job}__preview.jpg",
            "review": out_dir / "review" / f"{job}__review.jpg", "log": out_dir / "logs" / f"{job}.json",
            "staging": out_dir / "ok" / ".staging" / f"{job}__masked.png",
            "staging_preview": out_dir / "ok" / ".staging" / f"{job}__preview.jpg",}


# ---------------------------------------------------------------- 描画
def _scaled(img, long_edge=1600):
    s = min(1.0, long_edge / max(img.shape[:2]))
    return (cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()), s


def _label(img, lines, color=(255, 255, 255)):
    y = 22
    for t in lines:
        t = t.encode("ascii", "replace").decode()
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 22


def draw_review(img, cands, ctx, reasons, proposal=None):
    """原本に検出結果（候補四辺・T/F/A・理由コード）を重ねる。マスク前の画像なので欄の中身が見える（取扱注意）。"""
    vis, s = _scaled(img)
    lw = max(2, int(max(vis.shape[:2]) / 500))
    for c in cands or []:
        cv2.polylines(vis, [np.round(np.asarray(c["quad"]) * s).astype(np.int32)], True, (0, 200, 0), lw, cv2.LINE_AA)
    P = ctx.get("P_t2o")
    cues = ctx.get("cues")
    if P is not None and cues:
        for key, col in (("T", (255, 128, 0)), ("F", (0, 200, 255)), ("A", (255, 0, 255))):
            r = cues.get(key)
            if r:
                cv2.polylines(vis, [np.round(apply_h(P, rect_poly(r)) * s).astype(np.int32)], True, col, lw, cv2.LINE_AA)
    if proposal is not None:
        cv2.polylines(vis, [np.round(np.asarray(proposal) * s).astype(np.int32)], True, (0, 0, 255), lw + 1, cv2.LINE_AA)
    codes = []
    for r in reasons:
        if r["code"] not in codes:
            codes.append(r["code"])
    prim = primary_reason(reasons)
    if prim in codes:  # 一次理由を先頭に（切り詰められても案内が残る）。残りは出現順
        codes.remove(prim)
        codes.insert(0, prim)
    _label(vis, ["REVIEW (not masked): " + " ".join(codes[:6])], (0, 255, 255))
    return vis


def draw_preview(masked, polygon, quad):
    """マスク後の画像に、マスク枠（赤）とカード四辺（緑）を重ねる。欄の中身は写らない。"""
    vis, s = _scaled(masked)
    lw = max(2, int(max(vis.shape[:2]) / 500))
    cv2.polylines(vis, [np.round(np.asarray(polygon) * s).astype(np.int32)], True, (0, 0, 255), lw, cv2.LINE_AA)
    cv2.polylines(vis, [np.round(np.asarray(quad) * s).astype(np.int32)], True, (0, 200, 0), max(1, lw // 2), cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------- 1件の処理
def process_file(path, out_dir, cfg, templates, tmpl_sha, fill="black", shadow=False):
    """(rec, halt)。halt=True は原本が変化した（V6 失敗）ためバッチ全体を止める合図。"""
    path, out_dir = Path(path), Path(out_dir)
    t0 = time.perf_counter()
    rec = new_rec(path, cfg, tmpl_sha)
    rec["nondeterministic"]["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fill_bgr = FILL_COLORS[fill]
    ctx, cands, img, halt = {}, [], None, False
    try:
        data = path.read_bytes()
    except OSError as e:
        rec["job_id"] = "unreadable_" + imgio.sha256_bytes(str(path.resolve()).encode())[:8]
        rec["status"] = "error"
        rec["reasons"] = [{"code": "X_IO", "stage": "S0", "metric": type(e).__name__, "value": str(e)[:200],
                           "threshold": None}]
        return _finish(rec, out_dir, None, [], ctx, t0), False
    sha = imgio.sha256_bytes(data)
    job = sha[:16]
    rec["job_id"] = job
    before = imgio.file_state(path)
    rec["input"].update({"sha256": sha, "bytes": len(data), "mtime_ns": before["mtime_ns"]})
    pp = _out_paths(out_dir, job)
    placed = []
    if any(pp[k].exists() for k in ("masked", "preview", "review", "log")):  # 既存出力は上書きしない
        rec["status"] = "error"
        rec["reasons"] = [{"code": "X_OUTPUT_EXISTS", "stage": "S9", "metric": "exists", "value": True,
                           "threshold": False}]
        rec["_no_write"] = True
        rec["primary_reason"] = primary_reason(rec["reasons"])
        return rec, False

    try:
        # S0
        try:
            img, info = imgio.decode(data, int(cfg.input.max_pixels))
        except imgio.InputError as e:
            code = e.code
            rec["status"] = "error" if code == "E_DECODE" else "review"
            raise StageFail("S0", [{"code": code, "stage": "S0", "metric": "decode", "value": e.msg,
                                    "threshold": None}])
        rec["input"].update(info)
        long_edge = max(img.shape[:2])
        _add(rec, "S0", {"long_edge_px": long_edge})
        # S1
        res = _run("S1", s1_candidates, img, cfg, templates)
        cands = res["candidates"]
        _add(rec, "S1", s1_log(res))
        if not cands:
            rs = [{"code": "C_NO_CARD", "metric": "candidates", "value": 0, "threshold": 1}]
            for code, n in sorted(res["rejected"].items()):
                if code != "C_NOT_QUAD":
                    rs.append({"code": code, "metric": "rejected_candidates", "value": n, "threshold": 0})
            _fail("S1", rs)
        if len(cands) > cfg.card.max_candidates:
            _fail("S1", [{"code": "C_TOO_MANY_CANDIDATES", "metric": "candidates", "value": len(cands),
                          "threshold": cfg.card.max_candidates}])
        # S2 / S3
        cands_s = _run("S2", orient.refine_candidates, img, cands, templates, cfg)  # 精密化は S2〜S4 の照合・正規化だけ。ログと V4 は元の S1 の四角
        hyps = _run("S2", s2_score, img, cands_s, templates, cfg)
        top = sorted(hyps, key=lambda h: -h["zncc"])[:12]
        _add(rec, "S2", {"n_hypotheses": len(hyps), "top": [{k: h[k] for k in ("cand", "rot", "mirror", "template", "zncc")}
                                                            for h in top]})
        best, second, mirror, rs = _run("S3", s3_select, hyps, cands_s, cfg, templates)
        mirror, rs = apply_m3(img, cands_s, best, mirror, rs, templates, cfg)
        _add(rec, "S3", {"best": best, "second": second, "mirror": mirror})
        if rs:
            _fail("S3", rs)
        tmpl = next(t for t in templates if t.id == best["template"])
        quad = hypothesis_quads(cands_s[best["cand"]]["quad"])[best["rot"] // 90 + (4 if best["mirror"] else 0)][2]
        # S4
        norm, M = orient.normalize(img, quad, cfg)
        A, met, rs = _run("S4", s4_align, norm, tmpl, cfg)
        _add(rec, "S4", met)
        if rs or A is None:
            _fail("S4", rs)
        M10, A10 = np.array(sig12(M)), np.array(sig12(A))
        P = np.array(sig12(orient.template_to_orig(M10, A10)))
        ctx["P_t2o"] = P
        aligned = orient.aligned_image(img, P, cfg)
        # S5
        e = edge_lengths(quad)
        card_long = float(max((e[0] + e[2]) / 2, (e[1] + e[3]) / 2))
        met, rs = _run("S5", s5_quality, aligned, tmpl, card_long, cfg)
        if card_long < cfg.input.min_card_long_edge_px:
            rs = rs + [{"code": "Q_LOW_RES", "metric": "card_long_edge_px", "value": card_long,
                        "threshold": cfg.input.min_card_long_edge_px}]
        _add(rec, "S5", met)
        if rs:
            _fail("S5", rs)
        # S6
        cues, met, rs = _run("S6", s6_locate, aligned, tmpl, cfg)
        _add(rec, "S6", met)
        if rs or cues is None:
            _fail("S6", rs)
        ctx["cues"] = cues
        # S7
        rect, met, rs = _run("S7", s7_mask, aligned, tmpl, cues, cfg)
        _add(rec, "S7", met)
        if rs or rect is None:
            _fail("S7", rs)
        polygon = np.array(sig12(apply_h(P, rect_poly(rect))))
        geom = {"template_id": tmpl.id, "card_quad": order_quad(cands[best["cand"]]["quad"]),
                "card_color_fallback": any(str(m).startswith("lab_") for m in cands[best["cand"]]["maps"]),
                "M_n2o": M10, "ecc_A": A10, "P_t2o": P,
                "cues": {k: (None if cues.get(k) is None else cues[k]) for k in ("T", "F", "A", "A_boxes", "window", "R")},
                "mask_rect": [float(v) for v in rect], "mask_polygon": polygon, "fill_bgr": list(fill_bgr)}
        rec["geometry"] = sig12(geom)
        ctx["polygon"] = polygon
        # S8: 塗り → 検証（verify.py は原本・出力・ログの幾何だけを使う）
        g = rec["geometry"]
        m = verify.make_mask(img.shape, g["mask_polygon"], int(cfg.mask.grow_px))
        out = img.copy()
        out[m] = np.array(fill_bgr, np.uint8)
        vr = verify.verify_all(orig_arr=img, masked_arr=out, geom=g, tmpl=tmpl, cfg=cfg, fill_bgr=fill_bgr,
                               orig_path=path, orig_before=before)
        _add(rec, "S8", {k: vr[k] for k in verify.VERDICTS})
        if not vr["V6"]["pass"]:
            halt = True
        if not vr["pass"]:
            _fail("S8", verify.failed_reasons(vr))
        if shadow:
            _fail("S9", [{"code": "X_SHADOW", "metric": "shadow_mode", "value": True, "threshold": False}])
        # S9: masked と preview の両方を staging に書き、読み戻し検証(V2/V3)と V6・原本不変を
        #     すべて通過してから、まとめて ok/ へ確定する（review なら ok/ に1つも残さない）
        png = imgio.encode_png(out, int(cfg.output.png_compression))
        pv = imgio.encode_jpeg(draw_preview(out, polygon, order_quad(cands[best["cand"]]["quad"])),
                               int(cfg.output.preview_jpeg_quality))
        for k in ("staging", "staging_preview", "masked", "preview", "review"):
            pp[k].parent.mkdir(parents=True, exist_ok=True)
        stg, stg_pv = pp["staging"], pp["staging_preview"]
        try:
            stg.write_bytes(png)
            stg_pv.write_bytes(pv)
            back = imgio.read_array(stg)
            if back is None:
                raise OSError("読み戻せない")
            if imgio.read_array(stg_pv) is None:
                raise OSError("preview を読み戻せない")
            rb2, rb3 = verify.v2_uniform(back, m, fill_bgr), verify.v3_outside(back, img, m)
            _add(rec, "S9", {"readback_V2": rb2, "readback_V3": rb3})
            if not (rb2["pass"] and rb3["pass"]):
                _fail("S9", [{"code": c["code"], "metric": "readback", "value": False, "threshold": True}
                             for c in (rb2, rb3) if not c["pass"]])
            # 原本不変（V6）も ok/ へ出す前に確定させる（確定前チェックはここ 1 回だけ。末尾でもう一度確認する）
            v6 = verify.v6_original(path, before)
            rec["original_unchanged"] = bool(v6["pass"])
            if not v6["pass"]:
                halt = True
                _fail("S9", verify.failed_reasons({"V1": vr["V1"], "V2": vr["V2"], "V3": vr["V3"], "V4": vr["V4"],
                                                   "V5": vr["V5"], "V6": v6, "V7": vr["V7"]}))
            # v6_original が例外を投げた場合は original_unchanged が None のままなので、末尾の _check_original が拾う
            os.replace(stg, pp["masked"])
            placed.append("masked")
            os.replace(stg_pv, pp["preview"])
            placed.append("preview")
        finally:
            for s in (stg, stg_pv):
                try:
                    s.unlink(missing_ok=True)  # staging は納品対象ではない
                except OSError as e:  # 掃除の失敗で真因の理由コードを潰さない（外へは投げない）
                    rec["stages"]["S9_cleanup"] = {**rec["stages"].get("S9_cleanup", {}),
                                                   s.name: f"{type(e).__name__}: {str(e)[:120]}"}
        rec["status"] = "ok"
        rec["outputs"].update({"masked": str(pp["masked"].resolve()), "masked_sha256": imgio.sha256_bytes(png),
                               "preview": str(pp["preview"].resolve())})
    except StageFail as sf:
        if rec["status"] != "error":
            rec["status"] = "review"
        rec["reasons"] = [clean(r) for r in sf.reasons]
        halt = _abort_ok(pp, rec, placed) or halt
        if img is not None:
            _write_review(rec, pp, img, cands, ctx, sf.reasons, proposal=ctx.get("polygon") if shadow else None, cfg=cfg)
    except Exception as e:  # ここに来るのは段の外（描画・書き出し等）の想定外。ok にはしない
        rec["status"] = "review"
        rec["reasons"] = [{"code": "X_EXCEPTION", "stage": "S9", "metric": type(e).__name__,
                           "value": str(e)[:200], "threshold": None}]
        halt = _abort_ok(pp, rec, placed) or halt
        if img is not None:
            _write_review(rec, pp, img, cands, ctx, rec["reasons"], cfg=cfg)
    # 末尾でもう一度確認する（S9 で確定した後に原本が変わる競合を拾う）。既に False なら再読みしない
    if rec["original_unchanged"] is False or not _check_original(rec, path, before):
        halt = True
        if rec["status"] == "ok":  # 確定後に変わった。ok/ を残して review にしない
            rec["status"] = "review"
            rec["reasons"] = [{"code": "X_ORIGINAL_CHANGED", "stage": "S9", "metric": "sha256_mtime",
                               "value": False, "threshold": True}]
            halt = _abort_ok(pp, rec, placed) or halt
            if img is not None:
                _write_review(rec, pp, img, cands, ctx, rec["reasons"], cfg=cfg)
    return _finish(rec, out_dir, pp, cands, ctx, t0), halt


def _write_review(rec, pp, img, cands, ctx, reasons, proposal=None, cfg=None):
    try:
        vis = draw_review(img, cands, ctx, reasons, proposal)
        pp["review"].parent.mkdir(parents=True, exist_ok=True)
        pp["review"].write_bytes(imgio.encode_jpeg(vis, int(cfg.output.preview_jpeg_quality) if cfg else 90))
        rec["outputs"]["review"] = str(pp["review"].resolve())
    except Exception as e:  # review 画像が書けなくても判定は review のまま
        rec["reasons"].append({"code": "X_REVIEW_IMAGE_FAILED", "stage": "S9", "metric": type(e).__name__,
                               "value": str(e)[:200], "threshold": None})


def _check_original(rec, path, before):
    """原本の sha256/mtime が変わっていないか（判定は verify.v6_original に一本化）。rec に記録して bool を返す。"""
    try:
        rec["original_unchanged"] = bool(verify.v6_original(path, before)["pass"])
    except OSError:
        rec["original_unchanged"] = False
    return rec["original_unchanged"]


def _abort_ok(pp, rec, placed):
    """ok/ に確定した産物を取り消す。review/error に落とす経路は必ずここを通す（ok/ に残骸を残さない）。

    削除に失敗したら「ok/ に残骸があるまま review」にはせず、status=error・理由コードを残して
    halt=True を返しバッチを止める（残骸の存在を人が必ず気づく状態にする）。
    """
    failed = []
    for k in list(placed):
        try:
            if pp[k].exists():
                pp[k].unlink()
            placed.remove(k)
        except OSError as e:
            failed.append({"code": "X_OK_CLEANUP_FAILED", "stage": "S9", "metric": k,
                           "value": f"{type(e).__name__}: {str(e)[:160]}", "threshold": None})
    failed_keys = {f["metric"] for f in failed}
    for k in ("masked", "preview"):
        if k not in failed_keys:  # 削除できなかった分はパスを残す（ログを見れば残骸に気づける）
            rec["outputs"][k] = None
    if "masked" not in failed_keys:
        rec["outputs"]["masked_sha256"] = None
    if failed:
        rec["status"] = "error"
        rec["reasons"] = list(rec["reasons"]) + failed
        return True
    return False


def _finish(rec, out_dir, pp, cands, ctx, t0):
    # run・diagnostic・error の経路はここを通る（X_OUTPUT_EXISTS の早期 return は同じ式を直接入れている）。ok なら None
    rec["primary_reason"] = None if rec["status"] == "ok" else primary_reason(rec["reasons"])
    rec["nondeterministic"]["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    if rec.get("_no_write"):
        rec.pop("_no_write")
        return rec
    pp = pp or _out_paths(out_dir, rec["job_id"])
    pp["log"].parent.mkdir(parents=True, exist_ok=True)
    if pp["log"].exists():  # 念のため。上書きしない
        rec["status"], rec["reasons"] = "error", [{"code": "X_OUTPUT_EXISTS", "stage": "S9", "metric": "log",
                                                   "value": True, "threshold": False}]
        rec["primary_reason"] = primary_reason(rec["reasons"])
        return rec
    geom = rec.pop("geometry", None)
    rec = json.loads(json.dumps(clean(rec, 8), ensure_ascii=False))
    rec["geometry"] = sig12(geom) if geom is not None else None
    pp["log"].write_text(json.dumps(rec, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8")
    return rec


# ---------------------------------------------------------------- 診断（run --diagnostic。設計 §1.1a）
STAGE_NAMES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")
DIAG_NO_WRITE = "書き出しは run のみ"


def stage_summary(rec):
    """段ごとの 'pass' / 'fail' / 'skip'（ログの stages と reasons から導く。診断の1行表示用）。"""
    failed = {r.get("stage") for r in rec["reasons"]}
    out = {}
    for s in STAGE_NAMES:
        st = rec["stages"].get(s)
        if st is None:
            continue
        out[s] = "skip" if isinstance(st, dict) and "skipped" in st else ("fail" if s in failed else "pass")
    return out


def diagnose_file(path, out_dir, cfg, templates, tmpl_sha):
    """(rec, halt)。段の fail を記録するだけで先の段へ進み、可能な限り全段のメトリクスをログに残す。

    本番経路（process_file）は通らず、同じ段関数・検証関数を呼ぶ。status は常に review。ok/ にも review/ にも
    何も書かない（書くのは out/logs/<job_id>.json だけ）。前段の結果が無くて計算できない段は、値を推測せず
    {"skipped": "<前提段>"} を残す。理由コードは通常経路と同じ（落ちた段の理由が全部 rec.reasons に残る）。
    """
    path, out_dir = Path(path), Path(out_dir)
    t0 = time.perf_counter()
    rec = new_rec(path, cfg, tmpl_sha, mode="diagnostic")
    rec["nondeterministic"]["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reasons, halt = [], False

    def note(stage, rs):
        reasons.extend(clean(dict(r, stage=stage)) for r in rs)

    def call(stage, fn, *a):
        """段の呼び出し。想定外の例外は X_EXCEPTION として記録し、None を返して先へ進む。"""
        try:
            return _run(stage, fn, *a)
        except StageFail as sf:
            note(stage, sf.reasons)
            rec["stages"].setdefault(stage, {"failed": sf.reasons[0]["code"]})
            return None

    def skip(stages, because):
        for st in stages:
            _add(rec, st, {"skipped": because})

    try:
        data = path.read_bytes()
    except OSError as e:
        rec["job_id"] = "unreadable_" + imgio.sha256_bytes(str(path.resolve()).encode())[:8]
        rec["status"] = "review"
        rec["reasons"] = [{"code": "X_IO", "stage": "S0", "metric": type(e).__name__, "value": str(e)[:200],
                           "threshold": None}]
        return _finish(rec, out_dir, None, [], {}, t0), False
    sha = imgio.sha256_bytes(data)
    rec["job_id"] = sha[:16]
    before = imgio.file_state(path)
    rec["input"].update({"sha256": sha, "bytes": len(data), "mtime_ns": before["mtime_ns"]})
    pp = _out_paths(out_dir, rec["job_id"])
    if pp["log"].exists():  # 既存ログは上書きしない（job_id は通常 run と同じ。別の出力先を使う）
        rec["status"] = "error"
        rec["reasons"] = [{"code": "X_OUTPUT_EXISTS", "stage": "S9", "metric": "exists", "value": True,
                           "threshold": False}]
        rec["_no_write"] = True
        rec["primary_reason"] = primary_reason(rec["reasons"])
        return rec, False

    got = {}  # S8 に渡すもの: img / tmpl / rect / geom

    def stages():
        """S0〜S7。前提が揃わない段は skipped を残して return（以降の段も連鎖して skipped）。"""
        # S0
        try:
            img, info = imgio.decode(data, int(cfg.input.max_pixels))
        except imgio.InputError as e:
            note("S0", [{"code": e.code, "metric": "decode", "value": e.msg, "threshold": None}])
            _add(rec, "S0", {"failed": e.code})
            return skip(("S1", "S2", "S3", "S4", "S5", "S6", "S7"), "S0")
        got["img"] = img
        rec["input"].update(info)
        long_edge = max(img.shape[:2])
        _add(rec, "S0", {"long_edge_px": long_edge})
        # S1
        res = call("S1", s1_candidates, img, cfg, templates)
        cands = res["candidates"] if res else []
        if res:
            _add(rec, "S1", s1_log(res))
            if not cands:
                rs = [{"code": "C_NO_CARD", "metric": "candidates", "value": 0, "threshold": 1}]
                for code, n in sorted(res["rejected"].items()):
                    if code != "C_NOT_QUAD":
                        rs.append({"code": code, "metric": "rejected_candidates", "value": n, "threshold": 0})
                note("S1", rs)
            elif len(cands) > cfg.card.max_candidates:
                note("S1", [{"code": "C_TOO_MANY_CANDIDATES", "metric": "candidates", "value": len(cands),
                             "threshold": cfg.card.max_candidates}])
        if not cands:
            return skip(("S2", "S3", "S4", "S5", "S6", "S7"), "S1")
        # S2 / S3
        cands_s = call("S2", orient.refine_candidates, img, cands, templates, cfg) or cands
        hyps = call("S2", s2_score, img, cands_s, templates, cfg)
        if hyps is None:
            return skip(("S3", "S4", "S5", "S6", "S7"), "S2")
        top = sorted(hyps, key=lambda h: -h["zncc"])[:12]
        _add(rec, "S2", {"n_hypotheses": len(hyps),
                         "top": [{k: h[k] for k in ("cand", "rot", "mirror", "template", "zncc")} for h in top]})
        r3 = call("S3", s3_select, hyps, cands_s, cfg, templates)
        best = None
        if r3:
            best, second, mirror, rs = r3
            mirror, rs = apply_m3(img, cands_s, best, mirror, rs, templates, cfg)
            _add(rec, "S3", {"best": best, "second": second, "mirror": mirror})
            if rs:
                note("S3", rs)
        tmpl = next((t for t in templates if best is not None and t.id == best["template"]), None)
        if tmpl is None or not tmpl.positive:  # 対象様式（positive）を選べていない。S4 以降は意味を持たない
            return skip(("S4", "S5", "S6", "S7"), "S3")
        got["tmpl"] = tmpl
        quad = hypothesis_quads(cands_s[best["cand"]]["quad"])[best["rot"] // 90 + (4 if best["mirror"] else 0)][2]
        # S4
        nm = call("S4", orient.normalize, img, quad, cfg)
        r4 = call("S4", s4_align, nm[0], tmpl, cfg) if nm else None
        if r4:
            _add(rec, "S4", r4[1])
            if r4[2]:
                note("S4", r4[2])
        if r4 is None or r4[0] is None:
            return skip(("S5", "S6", "S7"), "S4")
        M10, A10 = np.array(sig12(nm[1])), np.array(sig12(r4[0]))
        P = np.array(sig12(orient.template_to_orig(M10, A10)))
        aligned = orient.aligned_image(img, P, cfg)
        # S5（落ちても S6 以降は計算する）
        e = edge_lengths(quad)
        card_long = float(max((e[0] + e[2]) / 2, (e[1] + e[3]) / 2))
        r5 = call("S5", s5_quality, aligned, tmpl, card_long, cfg)
        if r5:
            met, rs = r5
            if card_long < cfg.input.min_card_long_edge_px:
                rs = rs + [{"code": "Q_LOW_RES", "metric": "card_long_edge_px", "value": card_long,
                            "threshold": cfg.input.min_card_long_edge_px}]
            _add(rec, "S5", met)
            if rs:
                note("S5", rs)
        # S6
        r6 = call("S6", s6_locate, aligned, tmpl, cfg)
        if r6:
            _add(rec, "S6", r6[1])
            if r6[2]:
                note("S6", r6[2])
        if r6 is None or r6[0] is None:
            return skip(("S7",), "S6")
        cues = r6[0]
        # S7
        r7 = call("S7", s7_mask, aligned, tmpl, cues, cfg)
        if r7:
            _add(rec, "S7", r7[1])
            if r7[2]:
                note("S7", r7[2])
        if r7 is None or r7[0] is None:
            return
        rect = r7[0]
        polygon = np.array(sig12(apply_h(P, rect_poly(rect))))
        geom = {"template_id": tmpl.id, "card_quad": order_quad(cands[best["cand"]]["quad"]),
                "card_color_fallback": any(str(m).startswith("lab_") for m in cands[best["cand"]]["maps"]),
                "M_n2o": M10, "ecc_A": A10, "P_t2o": P,
                "cues": {k: (None if cues.get(k) is None else cues[k]) for k in ("T", "F", "A", "A_boxes", "window", "R")},
                "mask_rect": [float(v) for v in rect], "mask_polygon": polygon,
                "fill_bgr": list(FILL_COLORS["black"])}
        rec["geometry"] = sig12(geom)
        got["geom"] = rec["geometry"]

    def verify_stage():
        """S8（S7 の rect が取れたときだけ）。塗った画像はメモリ上だけで、どこにも書かない。"""
        img, g, tmpl = got["img"], got["geom"], got["tmpl"]
        m = verify.make_mask(img.shape, g["mask_polygon"], int(cfg.mask.grow_px))
        out = img.copy()
        out[m] = np.array(FILL_COLORS["black"], np.uint8)
        vr = verify.verify_all(orig_arr=img, masked_arr=out, geom=g, tmpl=tmpl, cfg=cfg,
                               fill_bgr=FILL_COLORS["black"], orig_path=path, orig_before=before)
        return vr

    try:
        stages()
        if got.get("geom") is None:
            skip(("S8",), "S7")
            skip(("S9",), "S8")
        else:
            vr = call("S8", verify_stage)
            if vr is not None:
                _add(rec, "S8", {k: vr[k] for k in verify.VERDICTS})
                if not vr["V6"]["pass"]:
                    halt = True
                if not vr["pass"]:
                    note("S8", verify.failed_reasons(vr))
            skip(("S9",), DIAG_NO_WRITE)
    except Exception as e:  # 診断自体の想定外。理由に残して review のまま（ok にはしない）
        note("S9", [{"code": "X_EXCEPTION", "metric": type(e).__name__, "value": str(e)[:200], "threshold": None}])
        for s in STAGE_NAMES:
            rec["stages"].setdefault(s, {"skipped": "X_EXCEPTION"})
    # 原本が変わっていないこと（診断も原本は読むだけ）
    if not _check_original(rec, path, before):
        halt = True
        note("S9", [{"code": "X_ORIGINAL_CHANGED", "metric": "sha256_mtime", "value": False, "threshold": True}])
    rec["status"] = "review"
    rec["reasons"] = reasons
    return _finish(rec, out_dir, pp, [], {}, t0), halt
