"""S1 カード候補の抽出。

面積では選ばない。幾何条件を満たす四角形は（入れ子を含め）全部候補として返し、
どれがカードかは S2/S3 の「罫線テンプレート照合」で決める。候補0個は C_NO_CARD（全体をカードとみなす経路は無い）。
"""
from __future__ import annotations

import cv2
import numpy as np

from . import orient
from .geom import (edge_lengths, interior_angles, order_quad, quad_area, quad_aspect, quad_diag, quad_dist)

MAP_ORDER = ("canny", "otsu", "otsu_inv")


def boundary_points(cn, hull, tol: float = 2.0) -> np.ndarray:
    """輪郭点のうち凸包の境界上（tol px 以内）にあるもの = カード外周の点。内側の罫線・文字のエッジを含まない。"""
    pts = cn.reshape(-1, 2)
    hp = hull.astype(np.float32)
    keep = [p for p in pts if abs(cv2.pointPolygonTest(hp, (float(p[0]), float(p[1])), True)) <= tol]
    return np.array(keep, np.float64).reshape(-1, 2)


def refine_corners(quad, edge_pts, band: float, min_pts: int):
    """各辺の近傍帯にある外周点へ fitLine(DIST_HUBER) を当て、隣り合う直線の交点を角にする。

    角丸カードで approxPolyDP の頂点が弧の上にずれるのを直す。辺の両端 10% は角丸の弧を避けて使わない。
    """
    lines = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        d = q - p
        L = float(np.linalg.norm(d))
        if L < 1:
            return quad, False
        u = d / L
        n = np.array([-u[1], u[0]])
        rel = edge_pts - p
        t, dist = rel @ u, np.abs(rel @ n)
        sel = (dist <= band) & (t >= 0.10 * L) & (t <= 0.90 * L)
        pts = edge_pts[sel]
        if len(pts) < min_pts:
            return quad, False
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = []
    for i in range(4):
        a, b = lines[(i - 1) % 4], lines[i]
        M = np.array([a[1], -b[1]]).T
        if abs(np.linalg.det(M)) < 1e-6:
            return quad, False
        t = np.linalg.solve(M, b[0] - a[0])
        out.append(a[0] + t[0] * a[1])
    new = np.array(out)
    if np.max(np.linalg.norm(new - quad, axis=1)) > band * 3:
        return quad, False
    return new, True


def edge_support(quad, dist_to_edge, tol: float) -> float:
    """四辺上の点のうち、tol px 以内に Canny エッジがある割合。"""
    h, w = dist_to_edge.shape
    vals = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        n = max(2, int(np.linalg.norm(q - p) / 2))
        for t in np.linspace(0, 1, n):
            x, y = p + t * (q - p)
            xi, yi = int(round(x)), int(round(y))
            vals.append(dist_to_edge[yi, xi] <= tol if 0 <= xi < w and 0 <= yi < h else False)
    return float(np.mean(vals)) if vals else 0.0


def is_frame_outline(quad, shape, cfg) -> bool:
    """外枠ゲート: 画像全体の輪郭がカード比の四角形か（C_FRAME_FILLED の前置き。これだけでは判定しない）。

    4角が画像の4隅から frame_filled_corner_px 以内にあり、見かけのアスペクトが
    カード比 1.585 ± frame_filled_aspect_tol のときだけ True。アスペクトだけではどの写真の
    外枠にも当たるので、カードが枠いっぱいかどうかは is_frame_filled の罫線照合で決める。
    """
    c = cfg.card
    h, w = shape[:2]
    tol = float(c.frame_filled_corner_px)
    q = order_quad(quad)
    corners = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])
    if float(np.linalg.norm(q - corners, axis=1).max()) > tol:
        return False
    asp = quad_aspect(q)
    return abs(asp - cfg.canvas.card_aspect) <= cfg.canvas.card_aspect * float(c.frame_filled_aspect_tol)


def is_frame_filled(img, templates, cfg) -> dict:
    """カードが枠いっぱいか: 画像全体を1枚のカードとみなした罫線照合（orient.frame_match）に status を付けて返す。

    status は "filled" | "not_filled" | "no_templates"。判定は理由コードとログにしか使わず、候補には足さない。
    """
    if not templates:
        return {"status": "no_templates"}
    m = orient.frame_match(img, templates, cfg)
    return dict(m, status="filled" if m["filled"] else "not_filled")


# 一次理由の優先順。撮り方で直せる「カード外形そのもの」の失敗を先にし、内部の区画や画像の縁から出やすい
# コード（C_ASPECT・C_AREA）を後にする。C_NOT_QUAD と表外のコードは対象外。
REJECT_PRIORITY = ("C_TOUCHES_BORDER", "C_FRAME_FILLED", "C_PERSPECTIVE", "C_ANGLE", "C_EDGE_SUPPORT",
                   "C_NO_CONSENSUS", "C_ASPECT", "C_AREA", "C_FRAME_UNVERIFIED")


def primary_rejection(rejected: dict) -> str | None:
    """rejected のうち REJECT_PRIORITY の順で最初に含まれるコード（件数は見ない）。無ければ None。"""
    for code in REJECT_PRIORITY:
        if rejected.get(code):
            return code
    return None


def geometry_check(quad, shape, dist_to_edge, cfg):
    """幾何条件（全部AND）。(失敗コード|None, メトリクス)。コードの優先順は area→angle→perspective→border→aspect→edge。"""
    c = cfg.card
    h, w = shape[:2]
    m = {}
    area_ratio = quad_area(quad) / float(h * w)
    m["area_ratio"] = area_ratio
    if not (c.min_area_ratio <= area_ratio < c.max_area_ratio):
        return "C_AREA", m
    ang = interior_angles(quad)
    m["angle_min"], m["angle_max"] = float(ang.min()), float(ang.max())
    if ang.min() < c.angle_min_deg or ang.max() > c.angle_max_deg:
        return "C_ANGLE", m
    e = edge_lengths(quad)
    r_tb = max(e[0], e[2]) / max(min(e[0], e[2]), 1e-9)
    r_lr = max(e[1], e[3]) / max(min(e[1], e[3]), 1e-9)
    m["opposite_ratio"] = max(r_tb, r_lr)
    if m["opposite_ratio"] > c.max_opposite_ratio:
        return "C_PERSPECTIVE", m
    mg = max(c.border_margin_px, c.border_margin_ratio * min(h, w))
    m["border_margin_px"] = float(min(quad[:, 0].min(), quad[:, 1].min(), w - 1 - quad[:, 0].max(),
                                      h - 1 - quad[:, 1].max()))
    if m["border_margin_px"] < mg:
        return "C_TOUCHES_BORDER", m
    asp = quad_aspect(quad)
    m["aspect"] = asp
    if abs(asp - cfg.canvas.card_aspect) > cfg.canvas.card_aspect * c.aspect_tol:
        return "C_ASPECT", m
    sup = edge_support(quad, dist_to_edge, c.edge_support_tol_px)
    m["edge_support"] = sup
    if sup < c.edge_support_min:
        return "C_EDGE_SUPPORT", m
    return None, m


def find_candidates(img, cfg, templates) -> dict:
    """{"candidates":[{"quad":(4,2) 元画像座標・時計回り, "maps":[...], "metrics":{...}}], "rejected":{code:n},
    "frame_filled": {...}, ...}

    templates は必須（既定値なし）。枠いっぱいの判定に使う。None は「照合しない」と明示する値で、
    外枠ゲートに当たって候補が0件なら C_FRAME_UNVERIFIED になる（黙って無判定にしない）。
    """
    c = cfg.card
    H, W = img.shape[:2]
    s = min(1.0, c.work_long_edge / max(H, W))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, int(c.canny_low), int(c.canny_high))
    k = int(c.canny_dilate)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maps = [("canny", cv2.dilate(edges, np.ones((k, k), np.uint8))), ("otsu", th), ("otsu_inv", 255 - th)]
    dist_to_edge = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    img_area = float(sh * sw)

    passed, rejected = [], {}

    def rej(code):
        rejected[code] = rejected.get(code, 0) + 1

    n_outline = 0
    for name, mp in maps:
        cnts, _ = cv2.findContours(mp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cn in cnts:
            if cv2.contourArea(cn) < c.min_area_ratio * img_area * 0.8:
                continue
            hull = cv2.convexHull(cn)
            peri = cv2.arcLength(hull, True)
            quad = None
            for eps in c.approx_eps:
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    quad = ap.reshape(4, 2).astype(np.float64)
                    break
            if quad is None:
                rej("C_NOT_QUAD")
                continue
            quad = order_quad(quad)
            band = c.corner_refine_band * quad_diag(quad)
            quad, _refined = refine_corners(quad, boundary_points(cn, hull), band, int(c.corner_refine_min_points))
            quad = order_quad(quad)
            if is_frame_outline(quad, small.shape, cfg):
                # 画像全体の輪郭。候補にも幾何判定にも回さず数だけ数える。候補が1つも取れなかったときだけ、
                # 下で罫線照合して C_FRAME_FILLED にするか決める（アスペクトだけでは判定しない）
                n_outline += 1
                continue
            code, met = geometry_check(quad, small.shape, dist_to_edge, cfg)
            if code:
                rej(code)
                continue
            passed.append((name, quad, met))

    # マップ間の合意: 同じ四角形（4角の距離が対角線長の consensus_dist_ratio 以内）が複数マップで見つかること
    clusters = []
    for name, quad, met in passed:
        for cl in clusters:
            if quad_dist(cl["quad0"], quad) <= c.consensus_dist_ratio * quad_diag(quad):
                cl["members"].append((name, quad, met))
                break
        else:
            clusters.append({"quad0": quad, "members": [(name, quad, met)]})
    cands = []
    for cl in clusters:
        names = sorted({m[0] for m in cl["members"]}, key=MAP_ORDER.index)
        if len(names) < c.consensus_min_maps:
            rej("C_NO_CONSENSUS")
            continue
        base = cl["members"][0][1]
        aligned = []
        for _n, q, _m in cl["members"]:
            k0 = int(np.argmin([np.linalg.norm(np.roll(q, -r, axis=0) - base, axis=1).max() for r in range(4)]))
            aligned.append(np.roll(q, -k0, axis=0))
        quad = order_quad(np.mean(aligned, axis=0) / s)  # 元画像スケールへ
        met = dict(cl["members"][0][2])
        met["maps"] = names
        cands.append({"quad": quad, "maps": names, "metrics": met})
    cands.sort(key=lambda d: (round(float(d["quad"][:, 1].mean()), 0), round(float(d["quad"][:, 0].mean()), 0)))
    if n_outline == 0:
        frame = {"status": "no_outline"}
    elif cands:
        frame = {"status": "has_candidates", "n_outline": n_outline}  # 候補があれば照合しない（コストを使わない）
    else:
        frame = dict(is_frame_filled(img, templates, cfg), n_outline=n_outline)
        if frame["status"] == "filled":
            rej("C_FRAME_FILLED")  # 「カードが枠いっぱい」＝再撮影（余白を残す）を案内する。フォールバックはしない
        elif frame["status"] == "no_templates":
            rej("C_FRAME_UNVERIFIED")  # 照合できなかった。黙って True にも False にもしない
        # not_filled: 外枠は画像の縁でカードではない。コードは付けず、根拠は frame に残る
    return {"candidates": cands, "rejected": rejected, "work_scale": s, "n_passed_raw": len(passed),
            "frame_filled": frame}
