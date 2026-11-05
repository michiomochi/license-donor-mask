"""S6 欄の特定: 独立した3つの手がかり T（テンプレート位置）/ F（罫線の枠）/ A（選択肢の印の3連）。

全て位置合わせ済みのテンプレート座標系（1011x638）で行う。手がかり同士が一致しなければ review。
アンカーの種類は template.anchor_type で差し替える（box3 = 四角が縦に3つ / circle3 = 番号を○で囲む様式の候補）。
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

from . import layout
from .geom import rect_area, rect_clip, rect_inside, rect_iou, rect_union_bbox
from .template import Template, window_rect


# ------------------------------------------------------------------ F: 罫線の枠
def frame_support(rul_comp: np.ndarray, rect, band: int) -> float:
    """矩形の周長のうち、罫線画素が band px 以内に乗っている割合。"""
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    H, W = rul_comp.shape
    hits = tot = 0
    for (a, b, horiz) in ((y0, (x0, x1), True), (y1 - 1, (x0, x1), True), (x0, (y0, y1), False), (x1 - 1, (y0, y1), False)):
        for t in range(b[0], b[1]):
            if horiz:
                lo, hi = max(0, a - band), min(H, a + band + 1)
                seg = rul_comp[lo:hi, t] if 0 <= t < W else np.zeros(1)
            else:
                lo, hi = max(0, a - band), min(W, a + band + 1)
                seg = rul_comp[t, lo:hi] if 0 <= t < H else np.zeros(1)
            tot += 1
            hits += bool(np.any(seg))
    return hits / tot if tot else 0.0


def detect_frame(rul: np.ndarray, T, cfg):
    """T と IoU 最大の罫線連結成分の外接矩形。(rect|None, metrics)。"""
    f = cfg.field
    d = cv2.dilate(rul, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        pix = (lab == i) & (rul > 0)
        if not pix.any():
            continue
        ys, xs = np.nonzero(pix)
        rect = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        iou = rect_iou(rect, T)
        if iou >= f.frame_min_iou and (best is None or iou > best[0]):
            best = (iou, rect, i)
    if best is None:
        return None, {"found": False}
    iou, rect, i = best
    comp = ((lab == i) & (rul > 0)).astype(np.uint8)
    sup = frame_support(comp, rect, int(f.frame_support_band_px))
    return rect, {"found": True, "iou_T": iou, "support": sup}


def band_support(rul: np.ndarray, band, span, horizontal: bool, tol: int):
    """期待バンド [lo, hi]（横線なら y、縦線なら x）に罫線画素が乗っている割合。

    span は線に沿った範囲（横線なら x、縦線なら y）。band の外側 tol px まで許す。
    戻り: (支持率, 検出位置の中央値 px | None, ヒットした列/行の数)
    """
    H, W = rul.shape
    lo = max(0, int(round(band[0])) - int(tol))
    hi = min(H if horizontal else W, int(round(band[1])) + int(tol) + 1)
    s0 = max(0, int(round(span[0])))
    s1 = min(W if horizontal else H, int(round(span[1])))
    if hi <= lo or s1 <= s0:
        return 0.0, None, 0
    if horizontal:
        sub = rul[lo:hi, s0:s1] > 0          # (rows, cols)
        hit = sub.any(axis=0)
        pos = np.argmax(sub, axis=0) + lo
    else:
        sub = rul[s0:s1, lo:hi] > 0          # (rows, cols)
        hit = sub.any(axis=1)
        pos = np.argmax(sub, axis=1) + lo
    if not hit.size:
        return 0.0, None, 0
    n = int(hit.sum())
    return (n / float(hit.size)), (float(np.median(pos[hit])) if n else None), n


def detect_rule_segments(rul: np.ndarray, f_rules: dict, cfg):
    """§4 (D) の F 再定義: テンプレートの期待位置にある罫線（線分）の一致。

    閉じた矩形は要求しない（実様式に囲み枠が無いため）。横罫線・縦罫線のそれぞれで
    「期待バンドの範囲に罫線画素が乗っている列（行）の割合」＝支持率を測り、全部が
    `f_rules_support_min` 以上なら F 成立。F の矩形は宣言した期待位置の外接矩形
    （検出画素の外接にすると、1本外れた線が他の線の位置を押し広げるため）。

    戻り: (rect|None, metrics)。rect はテンプレート座標。
    """
    f = cfg.field
    tol = int(f.f_rules_tol_px)
    smin = float(f.f_rules_support_min)
    met = {"mode": "segments", "found": False, "support_min": smin, "tol_px": tol, "rules": []}
    xs, ys, ok = [], [], True
    for key, band_key, span_key, horizontal in (("h_lines", "y", "x", True), ("v_lines", "x", "y", False)):
        for i, rule in enumerate(f_rules.get(key) or []):
            band, span = rule[band_key], rule[span_key]
            sup, center, n = band_support(rul, band, span, horizontal, tol)
            found = sup >= smin
            met["rules"].append({"kind": "h" if horizontal else "v", "index": i,
                                 "band": [float(v) for v in band], "span": [float(v) for v in span],
                                 "support": float(sup), "center_px": center, "hit": n, "found": bool(found)})
            ok = ok and found
            if horizontal:
                ys += [float(band[0]), float(band[1])]
                xs += [float(span[0]), float(span[1])]
            else:
                xs += [float(band[0]), float(band[1])]
                ys += [float(span[0]), float(span[1])]
    if not met["rules"] or not ok:
        return None, met
    rect = (min(xs), min(ys), max(xs), max(ys))
    met.update({"found": True, "rect": list(rect)})
    return rect, met


# ------------------------------------------------------------------ A: アンカー（選択肢の印）
def single_candidates(dark_crop: np.ndarray, expected_side: float, anchor_type: str, cfg, origin):
    """窓内の暗画素マップの外側輪郭から、印の単体候補を取る。手書きで中が埋まっていても外側輪郭で判定する。"""
    f = cfg.field
    cnts, hier = cv2.findContours(dark_crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    ox, oy = origin
    for i, cn in enumerate(cnts):
        if hier[0][i][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(cn)
        if w < 3 or h < 3:
            continue
        side = (w + h) / 2.0
        if abs(side - expected_side) > f.anchor_side_tol * expected_side:
            continue
        if not (f.anchor_aspect_min <= w / h <= f.anchor_aspect_max):
            continue
        area = cv2.contourArea(cn)
        peri = cv2.arcLength(cn, True)
        if anchor_type == "box3":
            if len(cv2.approxPolyDP(cn, 0.04 * peri, True)) != 4:
                continue
            (_c, (rw, rh), _a) = cv2.minAreaRect(cn)
            if rw * rh <= 0 or area / (rw * rh) < f.anchor_rectangularity_min:
                continue
        elif anchor_type == "circle3":
            if peri <= 0 or 4 * np.pi * area / (peri * peri) < f.anchor_circularity_min:
                continue
        else:
            raise ValueError(f"未知の anchor_type: {anchor_type}")
        out.append({"cx": ox + x + w / 2.0, "cy": oy + y + h / 2.0, "w": float(w), "h": float(h),
                    "rect": (ox + x, oy + y, ox + x + w, oy + y + h)})
    out.sort(key=lambda d: (d["cy"], d["cx"]))
    return out


def find_triples(singles, expected_pitch: float, cfg):
    """縦に並ぶ同サイズの3連。条件: 辺長の変動係数・同一列・間隔の均一性・期待間隔。"""
    f = cfg.field
    res = []
    for combo in itertools.combinations(range(len(singles)), 3):
        b = [singles[i] for i in combo]
        sides = np.array([(d["w"] + d["h"]) / 2 for d in b])
        ms = float(sides.mean())
        if float(sides.std() / ms) > f.triple_side_cv_max:
            continue
        cx = np.array([d["cx"] for d in b])
        if float(np.abs(cx - cx.mean()).max()) > f.triple_xshift_max * ms:
            continue
        g1, g2 = b[1]["cy"] - b[0]["cy"], b[2]["cy"] - b[1]["cy"]
        if g1 <= 0 or g2 <= 0 or max(g1, g2) / min(g1, g2) > f.triple_gap_ratio_max:
            continue
        if any(abs(g - expected_pitch) > f.triple_pitch_tol * expected_pitch for g in (g1, g2)):
            continue
        res.append(b)
    return res


def anchor_field_estimate(triple, tmpl: Template):
    """3連の実位置から（平行移動＋等方スケールの）相似変換を求め、field_rect を写す。"""
    exp = np.array(tmpl.meta["anchor"]["centers"], np.float64)
    act = np.array([[d["cx"], d["cy"]] for d in triple])
    pe = float(np.mean(np.diff(exp[:, 1])))
    pa = float(np.mean(np.diff(act[:, 1])))
    s = pa / pe
    ce, ca = exp.mean(axis=0), act.mean(axis=0)
    x0, y0, x1, y1 = tmpl.rect("field_rect")
    pts = np.array([[x0, y0], [x1, y1]])
    m = ca + s * (pts - ce)
    return (float(m[0, 0]), float(m[0, 1]), float(m[1, 0]), float(m[1, 1])), s


# ------------------------------------------------------------------ 統合
def locate_field(aligned_bgr, tmpl: Template, cfg):
    """(cues|None, metrics, reasons)。cues は全て位置合わせ済みテンプレート座標の矩形。"""
    f = cfg.field
    dark = layout.dark_map(aligned_bgr, cfg)
    rul = layout.ruling_from_dark(dark, cfg)
    T = tmpl.rect("field_rect")
    win = window_rect(tmpl, cfg)
    req = list(tmpl.meta["required_cues"])
    met: dict = {"window": list(win), "required": req}
    reasons = []

    def fail(code, metric, value, thr):
        reasons.append({"code": code, "metric": metric, "value": value, "threshold": thr})

    cues = {"T": T, "window": win, "F": None, "A_boxes": None, "A": None}
    # F: テンプレートが f_rules を持つ場合は「線分の一致」（実様式用。§4 (D)）。
    #    持たない場合は従来どおり「T と IoU 最大の閉じた枠」（合成テンプレート用）
    f_rules = tmpl.meta.get("f_rules")
    if f_rules:
        F, fm = detect_rule_segments(rul, f_rules, cfg)
        met["F"] = fm
        if "F" in req:
            if F is None:
                worst = min(fm["rules"], key=lambda r: r["support"]) if fm["rules"] else None
                fail("F_FRAME_NOT_FOUND", "f_rule_support",
                     worst["support"] if worst else 0.0, f.f_rules_support_min)
            else:
                # 罫線は欄の外へも伸びる（上の区切り罫線は protected_rects の内側、右上の縦罫線は欄の外）。
                # 線分モードの F の矩形は「宣言した期待位置の外接」＝画像から測った欄ではないので、
                # 欄矩形 R と V1 には欄（T）の内側だけを使う。欄の外の線分で欄を広げない。
                fm["rect_declared"] = [float(v) for v in F]
                clipped = rect_clip(F, T)
                if rect_area(clipped) <= 0:
                    fail("F_CUE_DISAGREE", "F_rect_vs_field_rect", 0, 1)
                else:
                    cues["F"] = clipped
                    fm["rect_used"] = [float(v) for v in clipped]
    else:
        F, fm = detect_frame(rul, T, cfg)
        met["F"] = fm
        if "F" in req:
            if F is None:
                fail("F_FRAME_NOT_FOUND", "iou_T_best", 0, f.frame_min_iou)
            elif fm["support"] < f.frame_support_min:
                fail("F_FRAME_NOT_FOUND", "frame_support", fm["support"], f.frame_support_min)
            else:
                cues["F"] = F
    # M3（§1.5c の鏡像ゲート。印刷数字の左右反転照合）は v2.1 の一部未実装。
    #   黙って飛ばさず、判定を保留したことと、保留している理由コードをログに残す。
    met["M3"] = {"mode": str(cfg.select.mirror_m3),
                 "pending_code": "F_ANCHOR_MIRRORED" if str(cfg.select.mirror_m3) == "unimplemented" else None}
    # A（印刷された選択肢マーク）: required_cues に A が無いテンプレート（実様式に印刷マークが無い場合）では実行しない
    anchor = tmpl.meta.get("anchor") or {}
    x0, y0, x1, y1 = [int(round(v)) for v in win]
    met["A"] = {"skipped": "A" not in req, "anchor_type": tmpl.meta["anchor_type"]}
    if "A" in req:
        singles = single_candidates(dark[y0:y1, x0:x1], float(anchor["side"]), tmpl.meta["anchor_type"], cfg, (x0, y0))
        met["A"]["singles"] = len(singles)
        if len(singles) > f.anchor_max_singles:
            fail("F_ANCHOR_MULTIPLE", "singles", len(singles), f.anchor_max_singles)
        else:
            triples = find_triples(singles, float(anchor["pitch"]), cfg)
            met["A"]["triples"] = len(triples)
            if len(triples) == 0:
                fail("F_ANCHOR_NOT_FOUND", "triples", 0, 1)
            elif len(triples) > 1:
                fail("F_ANCHOR_MULTIPLE", "triples", len(triples), 1)
            else:
                tr = triples[0]
                A_rect, sc = anchor_field_estimate(tr, tmpl)
                cues["A_boxes"] = [d["rect"] for d in tr]
                cues["A"] = A_rect
                met["A"]["scale"] = sc
    if reasons:
        return None, met, reasons

    # 一致判定（全部 AND）
    ious = {}
    # F が線分（f_rules）のときは IoU(T,F) の代わりに各線分の支持率・位置ずれで判定済み（§4 (D)）
    if cues["F"] is not None and not f_rules:
        ious["TF"] = rect_iou(T, cues["F"])
        if ious["TF"] < f.iou_tf_min:
            fail("F_CUE_DISAGREE", "iou_TF", ious["TF"], f.iou_tf_min)
    if cues["F"] is not None and cues["A"] is not None:
        ious["FA"] = rect_iou(cues["F"], cues["A"])
        if ious["FA"] < f.iou_fa_min:
            fail("F_CUE_DISAGREE", "iou_FA", ious["FA"], f.iou_fa_min)
    if cues["A"] is not None:
        ious["TA"] = rect_iou(T, cues["A"])
        if ious["TA"] < f.iou_ta_min:
            fail("F_CUE_DISAGREE", "iou_TA", ious["TA"], f.iou_ta_min)
    met["ious"] = ious
    for k in ("F", "A"):
        r = cues[k]
        if r is not None and not rect_inside(r, win):
            fail("F_CUE_DISAGREE", f"{k}_outside_window", list(r), list(win))
    if cues["A_boxes"] and not all(rect_inside(b, win) for b in cues["A_boxes"]):
        fail("F_CUE_DISAGREE", "A_boxes_outside_window", 1, 0)
    if reasons:
        return None, met, reasons
    present = [cues[k] for k in ("T", "F", "A") if cues[k] is not None]
    cues["R"] = rect_union_bbox(present)  # 欄矩形 R = T ∪ F ∪ A の外接矩形（保守側）
    return cues, met, reasons
