"""S2 仮説生成と照合 / S3 一意性判定 / S4 精密位置合わせ（ECC）。"""
from __future__ import annotations

import cv2
import numpy as np

from . import layout
from .geom import affine3, canon_matrix, hypothesis_quads, quad_iou, warp_to_frame


def normalize(img, quad, cfg):
    """quad(4角, 向き付き) を 1011x638 へ。(正規化画像, 正規化->元画像の射影行列)"""
    W, H = int(cfg.canvas.width), int(cfg.canvas.height)
    M = canon_matrix(quad, W, H)
    return warp_to_frame(img, M, W, H), M


def score_hypotheses(img, candidates, templates, cfg) -> list:
    """候補 x 8仮説 x テンプレート の全組合せの ZNCC。固定順序で評価する。"""
    out = []
    for ci, cand in enumerate(candidates):
        for rot, mirrored, q in hypothesis_quads(cand["quad"]):
            norm, _ = normalize(img, q, cfg)
            rul = layout.ruling_map(norm, cfg)
            for t in templates:
                z = layout.zncc(layout.signature(rul, t.ignore_rects, cfg), t.sig)
                out.append({"cand": ci, "rot": rot, "mirror": mirrored, "template": t.id,
                            "kind": t.kind, "label": t.label, "zncc": z})
    return out


def select_unique(hyps, candidates, cfg):
    """S3。(best, second, reasons)。reasons は dict(code, metric, value, threshold) のリスト。"""
    s = cfg.select
    if not hyps:
        return None, None, [{"code": "O_LOW_SCORE", "metric": "hypotheses", "value": 0, "threshold": 1}]
    order = sorted(range(len(hyps)), key=lambda i: (-hyps[i]["zncc"], hyps[i]["cand"], hyps[i]["mirror"],
                                                    hyps[i]["rot"], hyps[i]["template"]))
    best = hyps[order[0]]
    # best と「候補・回転・鏡像・テンプレートのどれかが異なる」仮説 = best 以外の全て。その最大が second。
    second = hyps[order[1]] if len(order) > 1 else None
    reasons = []
    if best["kind"] != "positive":
        code = "O_FRONT_SIDE" if best["label"] == "front" else "O_UNSUPPORTED_CARD"
        reasons.append({"code": code, "metric": "best_template", "value": best["template"], "threshold": "positive"})
    if best["zncc"] < s.min_zncc:
        reasons.append({"code": "O_LOW_SCORE", "metric": "zncc", "value": best["zncc"], "threshold": s.min_zncc})
    if second is not None:
        margin = best["zncc"] - second["zncc"]
        if margin < s.min_margin or margin <= s.tie_eps:
            reasons.append({"code": "O_AMBIGUOUS", "metric": "zncc_margin", "value": margin,
                            "threshold": s.min_margin})
    if best["mirror"]:
        reasons.append({"code": "O_MIRRORED", "metric": "mirrored", "value": True, "threshold": False})
    for ci in range(len(candidates)):
        if ci == best["cand"]:
            continue
        pos = [h["zncc"] for h in hyps if h["cand"] == ci and h["kind"] == "positive"]
        if pos and max(pos) >= s.min_zncc and \
                quad_iou(candidates[ci]["quad"], candidates[best["cand"]]["quad"]) < s.multi_card_iou:
            reasons.append({"code": "C_MULTIPLE_CARDS", "metric": "other_positive_zncc", "value": max(pos),
                            "threshold": s.min_zncc})
            break
    return best, second, reasons


def refine_alignment(norm, template, cfg):
    """S4。罫線マップ同士の ECC(MOTION_AFFINE)。(A 2x3 | None, metrics, reasons)。

    A: テンプレート座標 -> 正規化画像座標（findTransformECC の warpMatrix）。
    """
    e = cfg.ecc
    W, H = int(cfg.canvas.width), int(cfg.canvas.height)
    ign = template.ignore_rects
    inp = layout.zero_rects(layout.ruling_map(norm, cfg), ign)
    tpl = layout.zero_rects(template.ruling, ign)
    sg = float(e.blur_sigma)
    inp_f = cv2.GaussianBlur(inp.astype(np.float32) / 255.0, (0, 0), sg)
    tpl_f = cv2.GaussianBlur(tpl.astype(np.float32) / 255.0, (0, 0), sg)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, int(e.iterations), float(e.eps))
    try:
        cc, warp = cv2.findTransformECC(tpl_f, inp_f, warp, cv2.MOTION_AFFINE, crit, None, int(e.gauss_filt_size))
    except cv2.error as ex:
        return None, {"error": str(ex).splitlines()[0][:120]}, \
            [{"code": "R_ECC_FAIL", "metric": "ecc_exception", "value": 1, "threshold": 0}]
    A = warp.astype(np.float64)
    sx, sy = float(np.hypot(A[0, 0], A[1, 0])), float(np.hypot(A[0, 1], A[1, 1]))
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    tx, ty = float(A[0, 2]), float(A[1, 2])
    met = {"correlation": float(cc), "tx": tx, "ty": ty, "scale_x": sx, "scale_y": sy, "rotation_deg": rot}
    reasons = []

    def bad(metric, value, thr):
        reasons.append({"code": "R_ECC_FAIL", "metric": metric, "value": value, "threshold": thr})

    if cc < e.min_correlation:
        bad("correlation", cc, e.min_correlation)
    lim = e.max_translation_ratio * W
    if abs(tx) > lim or abs(ty) > lim:
        bad("translation", max(abs(tx), abs(ty)), lim)
    if not (e.scale_min <= sx <= e.scale_max and e.scale_min <= sy <= e.scale_max):
        bad("scale", [sx, sy], [e.scale_min, e.scale_max])
    if abs(rot) > e.max_rotation_deg:
        bad("rotation_deg", rot, e.max_rotation_deg)
    return A, met, reasons


def template_to_orig(M_n2o, A) -> np.ndarray:
    """テンプレート座標 -> 元画像座標 = (正規化->元) ∘ (テンプレート->正規化)。"""
    P = np.asarray(M_n2o, np.float64) @ affine3(A)
    return P / P[2, 2]


def aligned_image(img, P_t2o, cfg):
    """元画像を、位置合わせ済みのテンプレート座標系（1011x638）へ1回の補間で写す。"""
    return warp_to_frame(img, P_t2o, int(cfg.canvas.width), int(cfg.canvas.height))
