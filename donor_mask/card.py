"""S1 カード候補の抽出。

面積では選ばない。幾何条件を満たす四角形は（入れ子を含め）全部候補として返し、
どれがカードかは S2/S3 の「罫線テンプレート照合」で決める。候補0個は C_NO_CARD（全体をカードとみなす経路は無い）。
"""
from __future__ import annotations

import cv2
import numpy as np

from . import orient
from .geom import (edge_lengths, interior_angles, order_quad, quad_area, quad_aspect, quad_diag, quad_dist)

MAP_ORDER = ("canny", "otsu", "otsu_inv", "lab_a", "lab_a_inv", "lab_b", "lab_b_inv", "gc_a", "gc_b")


def grabcut_map(small, seed_frac: float, margin: float, iters: int, grabcut_long_edge: int = 600):
    """GrabCut の前景マスク（0/255）。画像の縁 margin（比）を確実な背景、中心の seed_frac（比）四方を確実な前景の種にして始める。

    背景が机・床・手などで、明度も色味も大津で1本の閾値では分けられない写真向け（フォールバック専用）。
    乱数を使わないので結果は毎回同じ。
    """
    h0, w0 = small.shape[:2]
    # 作業解像度は 1200px だと GrabCut が約1秒かかる。外形の取得には 600px で足りるので縮めて計算し、マスクを元の大きさへ戻す
    gs = min(1.0, float(grabcut_long_edge) / max(h0, w0))
    if gs < 1.0:
        small = cv2.resize(small, None, fx=gs, fy=gs, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    # 色の種類が少ない画像（単色・2色の絵・白紙の枠など）では GMM が退化して GrabCut が極端に遅くなる（実測で数十秒）。
    # 写真は量子化しても数百色ある。色が少なければ使わない（候補0件のまま = 従来どおり review）
    q = (small >> 3).astype(np.int32)
    if np.unique((q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]).size < 64:
        return np.zeros((h0, w0), np.uint8)
    # 量子化では見えない微小な退化を避けるため、乱数を使わない微小な模様を足す（結果は毎回同じ）
    yy, xx = np.indices(small.shape[:2])
    pat = (((xx * 73856093) ^ (yy * 19349663)) % 5 - 2).astype(np.int32)  # 乱数を使わない決まった模様（-2〜2）
    small = np.clip(small.astype(np.int32) + pat[..., None], 0, 255).astype(np.uint8)
    mx, my = max(1, int(w * margin)), max(1, int(h * margin))
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[my:h - my, mx:w - mx] = cv2.GC_PR_BGD  # 枠の内側は「たぶん背景」から始め、中心だけを前景の種にする
    cy0, cy1, cx0, cx1 = int(h * (0.5 - seed_frac / 2)), int(h * (0.5 + seed_frac / 2)), int(w * (0.5 - seed_frac / 2)), int(w * (0.5 + seed_frac / 2))
    mask[cy0:cy1, cx0:cx1] = cv2.GC_FGD  # ユーザーが撮った写真は、カードが画像の中心付近にある前提（運用前提）
    mask[my:h - my, mx:w - mx][mask[my:h - my, mx:w - mx] == cv2.GC_PR_BGD] = cv2.GC_PR_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(0)  # GrabCut の初期クラスタリング(kmeans)は OpenCV 内部の乱数を使う。呼ぶたびに結果が変わらないよう固定する
        cv2.grabCut(small, mask, None, bgd, fgd, int(iters), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # 縁のはみ出し（影・指・机の模様の取り込み）は凸包の四角形を傾ける。カードの縁より十分細いものだけを開処理で落とす
    r = max(3, int(round(min(h, w) * 0.025)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        fg = np.where(lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    return cv2.resize(fg, (w0, h0), interpolation=cv2.INTER_NEAREST) if gs < 1.0 else fg


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


_SIFT_CACHE: dict = {}


def sift_candidates(img, cfg, templates):
    """4段目のフォールバック: 特徴点（SIFT）の対応と RANSAC の射影で、テンプレートのカード全面が画像のどこに写るかを求める。

    学習なし（OpenCV の古典手法。MIDV の基準手法と同じ考え方）。罫線の照合が落ちる斜め・低解像度の写真向け。
    採用条件は全部AND: 陽性テンプレートごとに RANSAC のインライアが card.sift_min_inliers 以上、射影の向きが正（左右反転でない）、
    4角が正しい凸四角形、幾何ゲート（面積・角度・遠近・縁・縦横比。辺の支持だけは輪郭を使わないので除く）。
    返す候補は maps=["sift"] で、S2 以降（罫線の照合・S3 の向き・S4 の ECC・V1〜V7）は他の候補と同じに通る。
    戻りは (候補のリスト, ログ用の dict)。
    """
    c = cfg.card
    log = {"tried": 0, "best_inliers": 0}
    if not templates:
        return [], log
    h0, w0 = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sc = float(c.sift_work_long_edge) / max(h0, w0)  # 小さい画像は拡大、大きい画像は縮小して、特徴点の尺度をそろえる
    g = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
    sift = cv2.SIFT_create(int(c.sift_n_features))
    ki, di = sift.detectAndCompute(g, None)
    if di is None or len(ki) < 8:
        return [], log
    W, H = int(cfg.canvas.width), int(cfg.canvas.height)
    corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float64)
    cfg_g = cfg.override("card", edge_support_min=0.0, border_margin_px=0.0, border_margin_ratio=0.0)  # 縁に接してよい（外へはみ出す＝カードが切れているものは落ちる）
    out = []
    for t in templates:
        if not t.positive or t.print_map is None:
            continue
        if t.id not in _SIFT_CACHE:
            _SIFT_CACHE[t.id] = sift.detectAndCompute(255 - t.print_map, None)  # print_map は印刷=白
        kt, dt = _SIFT_CACHE[t.id]
        if dt is None or len(kt) < 8:
            continue
        log["tried"] += 1
        m = cv2.BFMatcher().knnMatch(dt, di, k=2)
        good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < float(c.sift_ratio) * b.distance]
        if len(good) < 8:
            continue
        cv2.setRNGSeed(0)  # RANSAC の乱数を固定（同じ入力は同じ結果）
        Hm, mask = cv2.findHomography(np.float32([kt[a.queryIdx].pt for a in good]),
                                      np.float32([ki[a.trainIdx].pt for a in good]), cv2.RANSAC, float(c.sift_ransac_px))
        n = int(mask.sum()) if mask is not None else 0
        log["best_inliers"] = max(log["best_inliers"], n)
        if Hm is None or n < int(c.sift_min_inliers) or np.linalg.det(Hm[:2, :2]) <= 0:
            continue
        pts = np.c_[corners, np.ones(4)] @ Hm.T
        q = (pts[:, :2] / pts[:, 2:3]) / sc
        if not np.all(np.isfinite(q)):
            continue
        code, met = geometry_check(q, img.shape, np.zeros(img.shape[:2], np.float32), cfg_g)
        if code is not None:
            log["rejected"] = code
            continue
        met = dict(met, sift_inliers=n, sift_good=len(good), sift_template=t.id)
        out.append({"quad": q, "maps": ["sift"], "metrics": met})
    return out, log


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
    img_area = float(sh * sw)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    # 色の地図（フォールバック用）: 白〜クリーム色のカードを白い背景に置くと明度ではほぼ分離できない（素材写真）。
    # 明度の3地図で候補が1つも取れなかったときだけ、Lab の a/b（色味）の大津2値化と色味の Canny を足してやり直す。
    # 先に取れた入力は今までと同じ経路。足した地図も同じ幾何ゲートと「2地図以上の一致」を通る。
    color_maps = []
    cedges = np.zeros_like(edges)
    for nm, ch in (("lab_a", lab[:, :, 1]), ("lab_b", lab[:, :, 2])):
        chb = cv2.GaussianBlur(ch, (5, 5), 0)
        _, cth = cv2.threshold(chb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        color_maps += [(nm, cth), (nm + "_inv", 255 - cth)]
        cedges |= cv2.Canny(chb, int(c.canny_color_low), int(c.canny_color_high))

    def scan(maps_, edge_map, cfg_):
        dist_to_edge = cv2.distanceTransform(255 - edge_map, cv2.DIST_L2, 3)
        passed, rejected = [], {}

        def rej(code):
            rejected[code] = rejected.get(code, 0) + 1

        n_outline = 0
        for name, mp in maps_:
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
                code, met = geometry_check(quad, small.shape, dist_to_edge, cfg_)
                if code:
                    rej(code)
                    continue
                passed.append((name, quad, met))
        return passed, rejected, n_outline

    def consensus(passed, rejected):
        # マップ間の合意: 同じ四角形（4角の距離が対角線長の consensus_dist_ratio 以内）が複数マップで見つかること
        clusters = []
        for name, quad, met in passed:
            for cl in clusters:
                if quad_dist(cl["quad0"], quad) <= c.consensus_dist_ratio * quad_diag(quad):
                    cl["members"].append((name, quad, met))
                    break
            else:
                clusters.append({"quad0": quad, "members": [(name, quad, met)]})
        out = []
        for cl in clusters:
            names = sorted({m[0] for m in cl["members"]}, key=MAP_ORDER.index)
            if len(names) < c.consensus_min_maps:
                rejected["C_NO_CONSENSUS"] = rejected.get("C_NO_CONSENSUS", 0) + 1
                continue
            base = cl["members"][0][1]
            aligned = []
            for _n, q, _m in cl["members"]:
                k0 = int(np.argmin([np.linalg.norm(np.roll(q, -r, axis=0) - base, axis=1).max() for r in range(4)]))
                aligned.append(np.roll(q, -k0, axis=0))
            quad = order_quad(np.mean(aligned, axis=0) / s)  # 元画像スケールへ
            met = dict(cl["members"][0][2])
            met["maps"] = names
            out.append({"quad": quad, "maps": names, "metrics": met})
        out.sort(key=lambda d: (round(float(d["quad"][:, 1].mean()), 0), round(float(d["quad"][:, 0].mean()), 0)))
        return out

    base_maps = list(maps)
    passed, rejected, n_outline = scan(base_maps, edges, cfg)
    cands = consensus(passed, rejected)
    color_pass, color_rejected = False, None
    if not cands:
        # フォールバック: 色の地図・色味の辺の支持を足してやり直す。辺の許容だけ edge_support_tol_color_px に広げる
        # （明度差が小さい辺は Canny の位置が揺れる）。edge_support_min と他の幾何ゲートは同じ
        color_pass = True
        cfg_c = cfg.override("card", edge_support_tol_px=float(c.edge_support_tol_color_px))
        passed2, rejected2, n_outline2 = scan(base_maps + color_maps, edges | cedges, cfg_c)
        cands2 = consensus(passed2, rejected2)
        if cands2:
            passed, rejected, n_outline, cands = passed2, rejected2, n_outline2, cands2
        else:
            # 足しても取れなかった入力の理由は、今までどおり明度の3地図の棄却だけで出す（色の地図の棄却は混ぜない。
            # 混ぜると小さなカード等の一次理由が C_AREA から C_PERSPECTIVE 等へ変わってしまう）。色側はログにだけ残す
            color_rejected = rejected2
    grabcut_pass, grabcut_rejected = False, None
    if not cands and int(c.grabcut_fallback) == 1:
        # 3段目のフォールバック: 背景が複雑（手持ち・机・床）で明度・色味の2値化では外形が取れないとき、GrabCut の前景を
        # 地図として足す。前景の種の大きさを変えた2つの前景（gc_a/gc_b）が同じ四角形を返したときだけ候補になる。
        # 幾何ゲート・辺の支持・S2/S3 の罫線照合・V1〜V7 は同じ（緩めない）
        grabcut_pass = True
        gmaps = [("gc_a", grabcut_map(small, float(c.grabcut_seed_a), float(c.grabcut_margin), int(c.grabcut_iters))),
                 ("gc_b", grabcut_map(small, float(c.grabcut_seed_b), float(c.grabcut_margin), int(c.grabcut_iters)))]
        cfg_g = cfg.override("card", edge_support_tol_px=float(c.edge_support_tol_color_px))
        passed3, rejected3, n_outline3 = scan(gmaps, edges | cedges, cfg_g)
        cands3 = consensus(passed3, rejected3)
        if cands3:
            passed, rejected, n_outline, cands = passed3, rejected3, n_outline3, cands3
        else:
            grabcut_rejected = rejected3
    sift_pass, sift_log = False, None
    if int(c.sift_fallback) >= 1 and templates and (not cands or int(c.sift_fallback) >= 2):
        # 1: 他の経路が0件のときだけ。2: SIFT の四角が取れたら、それに置き換える。3: 輪郭の候補に足す
        sift_pass = True
        sc, sift_log = sift_candidates(img, cfg, templates)
        if int(c.sift_fallback) == 2:  # 置き換え
            cands = sc or cands
        else:  # 3: 輪郭の候補に足す（S2 の罫線照合で最良の候補を選ぶ）。ほぼ同じ四角は足さない
            cands = list(cands) + [q for q in sc if all(quad_dist(q["quad"], x["quad"]) > 0.01 * quad_diag(q["quad"]) for x in cands)]
    if n_outline == 0 and not cands:
        frame = {"status": "no_outline"}
    elif cands:
        frame = {"status": "has_candidates", "n_outline": n_outline}  # 候補があれば照合しない（コストを使わない）
    else:
        frame = dict(is_frame_filled(img, templates, cfg), n_outline=n_outline)
        if frame["status"] == "filled":
            rejected["C_FRAME_FILLED"] = rejected.get("C_FRAME_FILLED", 0) + 1  # 「カードが枠いっぱい」＝再撮影（余白を残す）を案内する。フォールバックはしない
        elif frame["status"] == "no_templates":
            rejected["C_FRAME_UNVERIFIED"] = rejected.get("C_FRAME_UNVERIFIED", 0) + 1  # 照合できなかった。黙って True にも False にもしない
        # not_filled: 外枠は画像の縁でカードではない。コードは付けず、根拠は frame に残る
    return {"candidates": cands, "rejected": rejected, "work_scale": s, "n_passed_raw": len(passed),
            "frame_filled": frame, "color_fallback": color_pass, "color_rejected": color_rejected,
            "grabcut_fallback": grabcut_pass, "grabcut_rejected": grabcut_rejected,
            "sift_fallback": sift_pass, "sift": sift_log}
