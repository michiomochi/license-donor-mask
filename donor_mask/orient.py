"""S2 仮説生成と照合 / S3 一意性判定 / S4 精密位置合わせ（ECC）。"""
from __future__ import annotations

import cv2
import numpy as np

from . import layout
from .geom import affine3, canon_matrix, hypothesis_quads, quad_iou, rect_iou, warp_to_frame


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


def _rank(h):
    """固定順序（同点を作らない）。S3 の並びはこのキーの昇順。"""
    return (-h["zncc"], h["cand"], h["mirror"], h["rot"], h["template"])


def score_quad(img, quad, template, cfg) -> float:
    """向き付きの1つの四角を1つのテンプレートで照合した ZNCC（score_hypotheses の内側ループ1回分）。"""
    norm, _ = normalize(img, quad, cfg)
    rul = layout.ruling_map(norm, cfg)
    return layout.zncc(layout.signature(rul, template.ignore_rects, cfg), template.sig)


def frame_match(img, templates, cfg) -> dict:
    """画像全体を1枚のカードとみなした罫線照合（C_FRAME_FILLED の判定。判定自体は card.is_frame_filled）。

    四角は元画像のフル解像度の四隅に固定（精密化の結果に依存させない）。全テンプレート×8仮説の最大 ZNCC が
    frame_filled_zncc_min 以上で、かつ同じ仮説で枠を中心へ縮めた照合値（frame_filled_inset_scales）のどれよりも
    高いときだけ filled。枠いっぱいなら縮めると罫線がずれて下がり、余白付きカードなら縮めるほうが外形に合って上がる。
    """
    H, W = img.shape[:2]
    q = np.array([[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]])
    hyps = score_hypotheses(img, [{"quad": q}], templates, cfg)
    best = min(hyps, key=_rank)
    tmpl = next(t for t in templates if t.id == best["template"])
    idx = best["rot"] // 90 + (4 if best["mirror"] else 0)
    c = q.mean(axis=0)
    scales = [float(s) for s in cfg.card.frame_filled_inset_scales]
    insets = []
    for s in scales:
        hq = hypothesis_quads(c + s * (q - c))[idx][2]
        insets.append(score_quad(img, hq, tmpl, cfg))
    zmin = float(cfg.card.frame_filled_zncc_min)
    inset_max = max(insets)
    return {"zncc": best["zncc"], "template": best["template"], "rot": best["rot"], "mirror": best["mirror"],
            "inset_scales": scales, "inset_zncc": insets, "inset_max": inset_max, "zncc_min": zmin,
            "filled": bool(best["zncc"] >= zmin and best["zncc"] > inset_max)}


def _hyp_quad(candidates, h):
    """仮説の四角（向き付き）。hypothesis_quads の並びは rot/90 + mirror*4。"""
    return hypothesis_quads(candidates[h["cand"]]["quad"])[h["rot"] // 90 + (4 if h["mirror"] else 0)][2]


def _same_family(a: str, b: str, templates, iou_min: float) -> bool:
    """§1.5b の配置ファミリー: field_rect の IoU が iou_min 以上の陽性テンプレート同士。

    テンプレート一覧を渡されなかった場合（テストの部分仮説）は、同じ id だけを同族とみなす。
    """
    if a == b:
        return True
    tmap = {t.id: t for t in (templates or [])}
    ta, tb = tmap.get(a), tmap.get(b)
    if ta is None or tb is None:
        return False
    ra, rb = ta.meta.get("field_rect"), tb.meta.get("field_rect")
    if ra is None or rb is None:
        return False  # 陰性テンプレート（表面・別様式）は field_rect を持たない＝同族になりえない
    return rect_iou(tuple(ra), tuple(rb)) >= iou_min


def find_twin(hypotheses, best, candidates, cfg, templates=None):
    """§1.5c 鏡像双子: best と同じ候補・同じ回転で、正規化画像が左右反転になる仮説。

    - 上下反転（鏡像＋180°回転）は回転が違うので twin にならない（主マージンの比較対象に残す）
    - 照合相手は best と同じ配置ファミリー（§1.5b）のテンプレートに限る
    - 定義は回転番号ではなく幾何で確かめる: best の四角を左右反転した四角と一致すること
    戻りは ZNCC 最大の twin（無ければ None）。
    """
    fam_iou = float(cfg.select.family_iou)
    q_best = _hyp_quad(candidates, best)
    q_flip = q_best[[1, 0, 3, 2]]  # 左右反転
    twins = []
    for h in hypotheses:
        if h is best or h["cand"] != best["cand"] or h["rot"] != best["rot"] or h["mirror"] == best["mirror"]:
            continue
        if not _same_family(h["template"], best["template"], templates, fam_iou):
            continue
        if float(np.abs(_hyp_quad(candidates, h) - q_flip).max()) > 1e-3:
            continue  # 幾何が左右反転でないものは twin ではない（実装依存にしない）
        twins.append(h)
    return min(twins, key=_rank) if twins else None


def select_unique(hyps, candidates, cfg, templates=None):
    """S3。(best, second, mirror, reasons)。

    - second は twin を除いた仮説の最大値（§1.5b / §1.5c）
    - mirror は{"twin", "d_mirror", "m_mirror", "m3"}。M2（d_mirror ≥ mirror_margin）と
      M3（S6 で判定。v2.1 では未実装なので "unimplemented" を明示）を持つ
    """
    s = cfg.select
    if not hyps:
        return None, None, {"twin": None, "d_mirror": None}, [
            {"code": "O_LOW_SCORE", "metric": "hypotheses", "value": 0, "threshold": 1}]
    best = min(hyps, key=_rank)
    twin = find_twin(hyps, best, candidates, cfg, templates)
    twins = [twin] if twin is not None else []
    rest = [h for h in hyps if h is not best and all(h is not t for t in twins)]
    second = min(rest, key=_rank) if rest else None
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
    # M2（§1.5c）: twin との差。twin が無い（同じテンプレートの鏡像仮説が入力に無い）ときは判定しない
    mirror = {"twin": twin, "d_mirror": None, "m_mirror": float(s.mirror_margin), "m3": str(s.mirror_m3)}
    if twin is not None:
        mirror["d_mirror"] = float(best["zncc"] - twin["zncc"])
        if mirror["d_mirror"] < float(s.mirror_margin):
            reasons.append({"code": "O_MIRROR_AMBIGUOUS", "metric": "d_mirror", "value": mirror["d_mirror"],
                            "threshold": float(s.mirror_margin)})
    for ci in range(len(candidates)):
        if ci == best["cand"]:
            continue
        pos = [h["zncc"] for h in hyps if h["cand"] == ci and h["kind"] == "positive"]
        if pos and max(pos) >= s.min_zncc and \
                quad_iou(candidates[ci]["quad"], candidates[best["cand"]]["quad"]) < s.multi_card_iou:
            reasons.append({"code": "C_MULTIPLE_CARDS", "metric": "other_positive_zncc", "value": max(pos),
                            "threshold": s.min_zncc})
            break
    return best, second, mirror, reasons


def mirror_m3(norm, template, cfg) -> dict:
    """§1.5c M3: 印刷数字「2」「3」の左右反転照合。best の正規化画像で、数字パッチを通常の向きと左右反転の向きで探す。

    「2」「3」は左右非対称なので、正しい向きなら通常パッチの照合値が高く反転パッチは下がる（罫線は左右ほぼ対称で区別できない）。
    探索窓は数字の中心の周り ±m3_search_px（位置合わせ前なので少し広く取る）。両方の数字で
    通常 ≥ m3_min_score かつ（通常 − 反転）≥ m3_margin のときだけ passed。数字パッチを持たないテンプレートは判定不能（passed=False）。
    """
    a = template.meta.get("anchor") or {}
    patches = getattr(template, "digit_patches", None) or []
    if template.meta.get("anchor_type") != "digits123" or len(patches) != 3:
        return {"status": "not_applicable", "passed": False}
    s = cfg.select
    g = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY) if norm.ndim == 3 else norm
    r = int(s.m3_search_px)
    rows = []
    for i in (1, 2):
        p = np.asarray(patches[i], np.float32)
        h, w = p.shape
        cx, cy = a["centers"][i]
        x0, y0 = int(cx - w / 2 - r), int(cy - h / 2 - r)
        x1, y1 = int(cx + w / 2 + r), int(cy + h / 2 + r)
        crop = g[max(y0, 0):y1, max(x0, 0):x1].astype(np.float32)
        if crop.shape[0] < h or crop.shape[1] < w:
            return {"status": "out_of_range", "passed": False}
        normal = float(cv2.matchTemplate(crop, p, cv2.TM_CCOEFF_NORMED).max())
        flipped = float(cv2.matchTemplate(crop, np.ascontiguousarray(p[:, ::-1]), cv2.TM_CCOEFF_NORMED).max())
        rows.append({"digit": i + 1, "normal": normal, "flipped": flipped})
    ok = all(x["normal"] >= float(s.m3_min_score) and x["normal"] - x["flipped"] >= float(s.m3_margin) for x in rows)
    return {"status": "checked", "passed": bool(ok), "digits": rows, "min_score": float(s.m3_min_score),
            "margin": float(s.m3_margin)}


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
