"""S8 マスク後の自動検証（V1〜V7）。全部 AND。1つでも落ちたら ok/ に出さない。

独立性: mask.py / pipeline.py / orient.py（マスクを作る側）は import しない。入力は
「原本」「出力ファイル（または出力配列）」「ログに記録した幾何」とテンプレート・設定だけ。
V4 の再検出のため、検出側の部品（card / field / ink / layout / geom）だけを使う。
（設計書 3 章の「マスク処理側のモジュールを import しない」= マスク矩形の作成ロジックを共有しない、と解釈した。）
"""
from __future__ import annotations

import cv2
import numpy as np

from . import card as cardmod
from . import field as fieldmod
from . import imgio, layout
from . import ink as inkmod
from .geom import apply_h, polygon_mask, quad_diag, quad_dist, rect_inside, rect_poly, warp_to_frame
from .template import window_rect

# S8 の判定項目（順序はログ・理由の並び）。全部 AND で、1つでも落ちたら ok/ に出さない。
VERDICTS = ("V1", "V2", "V3", "V4", "V5", "V6", "V7")


def make_mask(shape, polygon, grow: int) -> np.ndarray:
    return polygon_mask(shape, polygon, grow)


def _inv(P):
    return np.linalg.inv(np.asarray(P, np.float64))


def v1_inclusion(geom: dict, tmpl, cfg):
    """位置正当性: T/F/A がマスク内側（正規化座標で 6px 以上）、field_rect を100%含み IoU 0.70 以上、F/A は探索窓内。"""
    v = cfg.verify
    P = np.asarray(geom["P_t2o"], np.float64)
    poly_o = np.asarray(geom["mask_polygon"], np.float64)
    poly_t = apply_h(_inv(P), poly_o)  # マスク多角形をテンプレート座標へ戻す
    contour = poly_t.astype(np.float32).reshape(-1, 1, 2)
    cues = geom["cues"]
    pts = {"T": rect_poly(cues["T"])}
    if cues.get("F"):
        pts["F"] = rect_poly(cues["F"])
    if cues.get("A_boxes"):
        allb = np.array(cues["A_boxes"], np.float64)
        pts["A"] = rect_poly((allb[:, 0].min(), allb[:, 1].min(), allb[:, 2].max(), allb[:, 3].max()))
    min_d, per = 1e9, {}
    for k, pp in pts.items():
        d = min(cv2.pointPolygonTest(contour, (float(x), float(y)), True) for x, y in pp)
        per[k] = d
        min_d = min(min_d, d)
    fr = tuple(float(x) for x in tmpl.meta["field_rect"])
    fr_in = all(cv2.pointPolygonTest(contour, (float(x), float(y)), True) >= -1e-6 for x, y in rect_poly(fr))
    # IoU（正規化座標で点描画）
    W, H = int(cfg.canvas.width), int(cfg.canvas.height)
    m_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m_mask, [np.round(poly_t).astype(np.int32)], 1)
    f_mask = np.zeros((H, W), np.uint8)
    f_mask[int(round(fr[1])):int(round(fr[3])), int(round(fr[0])):int(round(fr[2]))] = 1
    inter = float(np.count_nonzero(m_mask & f_mask))
    iou = inter / max(1.0, float(np.count_nonzero(m_mask | f_mask)))
    win = window_rect(tmpl, cfg)
    win_ok = True
    for k in ("F", "A"):
        r = cues.get(k)
        if r and not (r[0] >= win[0] - 1e-6 and r[1] >= win[1] - 1e-6 and r[2] <= win[2] + 1e-6 and r[3] <= win[3] + 1e-6):
            win_ok = False
    # マスク多角形が記録された mask_rect と整合していること
    poly_re = apply_h(P, rect_poly(geom["mask_rect"]))
    consistent = bool(np.max(np.linalg.norm(poly_re - poly_o, axis=1)) < 0.01)
    ok = (min_d >= v.inside_min_dist_px - 1e-6) and fr_in and iou >= v.field_iou_min and win_ok and consistent
    return {"pass": bool(ok), "code": None if ok else "V_INCLUSION",
            "metrics": {"min_inside_dist_norm_px": min_d, "per_cue": per, "contains_field_rect": fr_in,
                        "iou_field_rect": iou, "cues_in_window": win_ok, "polygon_consistent": consistent}}


def v2_uniform(arr, mask, fill_bgr):
    """マスク内の画素が全て塗り色と完全一致。"""
    if not mask.any():
        return {"pass": False, "code": "V_FILL_NOT_UNIFORM", "metrics": {"mask_pixels": 0}}
    region = arr[mask]
    bad = int(np.count_nonzero(np.any(region != np.array(fill_bgr, np.uint8), axis=1)))
    return {"pass": bad == 0, "code": None if bad == 0 else "V_FILL_NOT_UNIFORM",
            "metrics": {"mask_pixels": int(mask.sum()), "mismatch_pixels": bad}}


def v3_outside(arr, orig, mask):
    """マスク外の画素が原本（EXIF 適用後のデコード配列）と完全一致。"""
    if arr.shape != orig.shape:
        return {"pass": False, "code": "V_OUTSIDE_CHANGED", "metrics": {"shape_mismatch": True}}
    diff = np.any(arr != orig, axis=2) & ~mask
    n = int(np.count_nonzero(diff))
    return {"pass": n == 0, "code": None if n == 0 else "V_OUTSIDE_CHANGED", "metrics": {"diff_pixels_outside": n}}


def _digit_max_in_rects(gray: np.ndarray, patches, rects) -> tuple:
    """rects（テンプレート座標）のそれぞれでパッチを照合し、最大値と「照合できた矩形の数」を返す。

    矩形がパッチより小さい（＝そこに数字が収まらない）ときは照合しない。1つも照合できなければ
    最大値は -1 で、呼び出し側が fail-closed に倒す。
    """
    H, W = gray.shape[:2]
    best, n = -1.0, 0
    for rect in rects:
        x0, y0, x1, y1 = [int(round(c)) for c in rect]
        crop = gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)].astype(np.float32)
        for p in patches:
            if crop.shape[0] < p.shape[0] or crop.shape[1] < p.shape[1]:
                continue
            r = np.nan_to_num(cv2.matchTemplate(crop, p.astype(np.float32), cv2.TM_CCOEFF_NORMED), nan=-1.0)
            best = max(best, float(r.max()))
            n += 1
    return best, n


def v4_redetect(masked, geom: dict, tmpl, cfg, mask):
    """出力に S1 とアンカー検出を再適用し、痕跡が残っていないこと。"""
    v = cfg.verify
    res: dict = {"card": None, "singles": None, "triples": None, "edges_in_mask": None}
    # カード四辺が壊れていない
    logged = np.asarray(geom["card_quad"], np.float64)
    # 黒塗りで Otsu の閾値が変わり、マスクがカード縁に近いと縁のエッジも弱まる。再検出は 1 マップ合意・縁の支持率 0.5 で緩め、
    # 代わりに「ログの四角との角の一致 1%」を厳密に見る（カード幾何が壊れていないことの確認が目的）
    # templates=None: V4 が使うのは candidates だけ。枠いっぱいの照合（C_FRAME_FILLED の判定）は不要なので明示的に省く
    # 色味で見つけたカード（S1 のフォールバック）は辺の位置が数px揺れるので、許容だけ別キーで広げる
    ratio = v.card_corner_dist_ratio_color if geom.get("card_color_fallback") else v.card_corner_dist_ratio
    found = cardmod.find_candidates(masked, cfg.override("card", consensus_min_maps=1, edge_support_min=0.5, sift_fallback=0),
                                    None)["candidates"]
    dmin = min((quad_dist(logged, c["quad"]) for c in found), default=1e9)
    res["card"] = {"n_candidates": len(found), "min_corner_dist_px": dmin,
                   "limit_px": ratio * quad_diag(logged)}
    card_ok = dmin <= res["card"]["limit_px"]
    if int(v.v4_card_missing_ok) == 1 and len(found) == 0:
        card_ok = True  # 塗りがカードの縁へ多少はみ出して再検出できなくても許容する（塗り残しは V1〜V3・V5・V7 が見る）
    if geom.get("card_sift"):
        # 特徴点（SIFT）の射影で求めた四角は輪郭に基づかない。輪郭が取れない写真向けなので、再検出で輪郭が見つからなくても
        # 「カード四辺が壊れた」とは言えない。輪郭が見つかったときは、広めの許容（verify.card_corner_dist_ratio_sift）で一致を見る。
        # 塗り残し・はみ出しは V1〜V3・V5・V7 と下の点検（探索窓・マスク内エッジ）が引き続き見る
        res["card"]["sift"] = True
        res["card"]["limit_px"] = float(v.card_corner_dist_ratio_sift) * quad_diag(logged)
        card_ok = len(found) == 0 or dmin <= res["card"]["limit_px"]
    # 探索窓内にチェックボックス単体候補・3連が残っていない
    P = np.asarray(geom["P_t2o"], np.float64)
    W, H = int(cfg.canvas.width), int(cfg.canvas.height)
    al = warp_to_frame(masked, P, W, H)
    dark = layout.dark_map(al, cfg)
    x0, y0, x1, y1 = [int(round(c)) for c in window_rect(tmpl, cfg)]
    if tmpl.meta["anchor_type"] == "none":
        # 印刷された選択肢マークが実在しない様式。再検出する対象が無い（＝残っていない）
        singles, triples = [], []
        res["anchor"] = "none"
    elif tmpl.meta["anchor_type"] == "digits123":
        # 設計 §3 V4 の「数字パッチの照合 < 0.3」を digits123 で具体化したもの。測るのは
        # 「位置決めに使った数字の3矩形」そのもの（出力の画素を読む）。
        #   - 数字の3矩形がマスク矩形に完全に入っていること（塗り残しの検出。ここが破れる経路は実在する:
        #     S7 が保護領域との隙間で余白を縮めたときにマスクが欄の内側へ寄る場合）
        #   - その3矩形の中の照合最大値 < digit_absent_max（塗りが実際に効いていること）
        # 探索窓全体では測らない: 窓の外（保護領域や欄の外）には別の印刷数字が実在し、塗ってはいけないので、
        # そこで高い値が出るのは正常（偽の fail になる）。
        singles, triples = [], []
        res["anchor"] = "digits123"
        boxes = [tuple(float(c) for c in b) for b in (geom["cues"].get("A_boxes") or [])]
        mrect = tuple(float(c) for c in geom["mask_rect"])
        res["digit_boxes"] = len(boxes)
        res["digit_boxes_in_mask"] = bool(boxes) and all(rect_inside(b, mrect) for b in boxes)
        res["digit_max"], res["digit_regions"] = _digit_max_in_rects(layout.to_gray(al), tmpl.digit_patches, boxes)
    else:
        anchor = tmpl.meta["anchor"]
        singles = fieldmod.single_candidates(dark[y0:y1, x0:x1], float(anchor["side"]), tmpl.meta["anchor_type"],
                                             cfg, (x0, y0))
        triples = fieldmod.find_triples(singles, float(anchor["pitch"]), cfg)
    res["singles"], res["triples"] = len(singles), len(triples)
    # マスク内を侵食してから Canny エッジを数える
    e = int(v.redetect_erode_px)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * e + 1, 2 * e + 1), np.uint8)).astype(bool)
    edges = cv2.Canny(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), int(cfg.card.canny_low), int(cfg.card.canny_high))
    res["edges_in_mask"] = int(np.count_nonzero(edges[inner]))
    if not card_ok:
        return {"pass": False, "code": "V_CARD_DAMAGED", "metrics": res}
    ok = len(singles) == 0 and len(triples) == 0 and res["edges_in_mask"] == 0
    if "digit_max" in res:
        # 照合できた矩形が 0 なら「確かめられない」ので通さない（fail-closed）
        ok = (ok and res["digit_boxes"] > 0 and res["digit_boxes_in_mask"] and res["digit_regions"] > 0
              and res["digit_max"] < v.digit_absent_max)
    return {"pass": ok, "code": None if ok else "V_REDETECT_TRACE", "metrics": res}


def v5_boundary_ink(orig, geom: dict, tmpl, cfg):
    """最終マスク矩形の外周リングにインクが残っていない（原本を位置合わせして再計算）。"""
    P = np.asarray(geom["P_t2o"], np.float64)
    al = warp_to_frame(orig, P, int(cfg.canvas.width), int(cfg.canvas.height))
    ink = inkmod.ink_map(al, tmpl.print_map, cfg)
    bad, met = inkmod.check_boundary_ink(ink, geom["mask_rect"], cfg)
    return {"pass": not bad, "code": None if not bad else "V_BOUNDARY_INK",
            "metrics": {"bad_sides": sorted(bad), **met}}


def v6_original(path, before: dict):
    after = imgio.file_state(path)
    same = after["sha256"] == before["sha256"] and after["mtime_ns"] == before["mtime_ns"]
    return {"pass": same, "code": None if same else "X_ORIGINAL_CHANGED",
            "metrics": {"sha256_same": after["sha256"] == before["sha256"],
                        "mtime_same": after["mtime_ns"] == before["mtime_ns"]}}


def v7_ink_outside(orig, geom: dict, tmpl, cfg):
    """マスク外のカード全面でインクを探す（設計 §3 V7 / P1-L4）。

    インク（§2.3 の定義）＝暗画素 − print_map（空欄カードの暗画素を膨張）− 縁。ここから
    マスク内・保護領域内・ignore_rects 内の画素を落とし、残った連結成分のうち外接矩形の長辺が
    `ink_min_len_px` 以上で、カード縁 `ink_edge_margin_px` 以内・四隅 `ink_corner_ignore_px` 四方に
    かからないものがあれば
    V_INK_OUTSIDE（review）。V5 のリング（マスク外周 12px）より遠い追記・はみ出し・訂正線を拾う。

    四隅を除くのは、カードの角が丸い場合に「角の外側の明るい背景」と「カード本体」の明暗差が
    縁に沿った縞として暗画素になるため（実物の角の半径は約3mm≒35px、紙台紙では角の縁が長い縞になる）。
    手書きではないので数えない。**この除外は四隅 48px 四方を「見ない」ので、その範囲に書かれた
    手書きは拾えない**（要校正）。
    """
    k = cfg.ink
    W, H = int(cfg.canvas.width), int(cfg.canvas.height)
    P = np.asarray(geom["P_t2o"], np.float64)
    al = warp_to_frame(orig, P, W, H)
    ink = inkmod.ink_map(al, tmpl.print_map, cfg)
    # 見ない領域: マスク（塗る所）・保護領域（情報が乗る）・ignore_rects（正当に手書きが乗る欄）。
    # 成分を数える前に画素を落とす（領域をまたぐ長い線は、外側に出た部分だけで判定する）。
    excl = np.zeros((H, W), np.uint8)
    poly_t = apply_h(_inv(P), np.asarray(geom["mask_polygon"], np.float64))
    cv2.fillPoly(excl, [np.round(poly_t).astype(np.int32)], 1)
    for r in list(tmpl.ignore_rects) + list(tmpl.meta["protected_rects"]):
        cv2.rectangle(excl, (int(round(r[0])), int(round(r[1]))), (int(round(r[2])), int(round(r[3]))), 1, -1)
    stroke_found = _protected_strokes(ink, poly_t, tmpl, k, W, H)
    ink = np.where(excl > 0, 0, ink).astype(np.uint8)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), connectivity=8)
    lo, m = int(k.ink_min_len_px), int(k.ink_edge_margin_px)
    c = int(k.ink_corner_ignore_px)
    corners = [(0, 0, c, c), (W - c, 0, W, c), (0, H - c, c, H), (W - c, H - c, W, H)] if c > 0 else []
    found, longest = [], 0
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i][:4])
        long_px = max(w, h)
        longest = max(longest, long_px)
        if long_px < lo:
            continue
        zone = int(k.ink_thin_line_zone_px)
        near_edge = zone <= 0 or x < zone or y < zone or x + w > W - zone or y + h > H - zone
        if int(k.ink_thin_line_px) > 0 and min(w, h) <= int(k.ink_thin_line_px) and near_edge:
            continue  # 幅が数px以下の細長い線のうち、カード縁の近く（zone px 以内。0=範囲制限なし）のものは文字ではないので数えない（0=無効）。縁から離れた細い一画（ボールペンの線）は数える
        if x < m or y < m or x + w > W - m or y + h > H - m:  # カード縁の成分は数えない
            continue
        if c > 0 and any(x < cx1 and x + w > cx0 and y < cy1 and y + h > cy0
                        for cx0, cy0, cx1, cy1 in corners):
            continue  # 四隅（丸い角の外側の背景・紙の縁）の成分は数えない
        found.append({"bbox": [x, y, x + w, y + h], "long_px": long_px, "area": int(stats[i][4])})
    found.extend(stroke_found)
    found.sort(key=lambda f: -f["long_px"])
    ok = not found
    return {"pass": ok, "code": None if ok else "V_INK_OUTSIDE",
            "metrics": {"n_components": len(found), "longest_px": longest,
                        "min_len_px": lo, "edge_margin_px": m, "worst": found[:5]}}


def _protected_strokes(ink, poly_t, tmpl, k, W, H):
    """マスクから続いて保護領域へはみ出した長い一画を拾う（V7 は保護領域の画素を落とすので、ここで別に見る）。

    保護領域（備考欄など）には文字が正当に載るので、領域内の文字は数えない。数えるのは「マスクの近く
    （ring_px の約 2 倍以内）に接していて、マスクから protected_stroke_px 以上離れた所まで届く成分」だけ
    （文字は欄のすぐ上に収まるが、欄から続く一画はそれより遠くまで届く）。0 = 無効。向きには依存しない。
    限界: 線が罫線の帯などで途切れていると拾えない（つなげると備考欄の文字まで一塊になり、正常な画像を止めるため採らない）。
    """
    n_min = int(k.protected_stroke_px)
    if n_min <= 0:
        return []
    prot = np.zeros((H, W), np.uint8)
    for r in tmpl.meta["protected_rects"]:
        cv2.rectangle(prot, (int(round(r[0])), int(round(r[1]))), (int(round(r[2])), int(round(r[3]))), 1, -1)
    near = np.zeros((H, W), np.uint8)
    cv2.fillPoly(near, [np.round(poly_t).astype(np.int32)], 1)
    g = 2 * int(k.ring_px) + 1
    near = cv2.dilate(near, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
    inv = np.ones((H, W), np.uint8)
    cv2.fillPoly(inv, [np.round(poly_t).astype(np.int32)], 0)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    cand = ((ink > 0) & (prot > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = []
    for i in range(1, n):
        comp = lab == i
        x, y, w, h = (int(v) for v in stats[i][:4])
        reach = float(dist[comp].max())
        if reach < n_min or not (near[comp] > 0).any():
            continue
        out.append({"bbox": [x, y, x + w, y + h], "long_px": int(round(reach)), "area": int(stats[i][4]), "kind": "protected_stroke"})
    return out


def verify_all(*, orig_arr, masked_arr, geom, tmpl, cfg, fill_bgr, orig_path=None, orig_before=None):
    """V1〜V7 を全部実行して AND を返す（途中で止めず全項目の値を残す）。"""
    mask = make_mask(orig_arr.shape, geom["mask_polygon"], int(cfg.mask.grow_px))
    out = {"V1": v1_inclusion(geom, tmpl, cfg),
           "V2": v2_uniform(masked_arr, mask, fill_bgr),
           "V3": v3_outside(masked_arr, orig_arr, mask)}
    try:
        out["V4"] = v4_redetect(masked_arr, geom, tmpl, cfg, mask)
    except Exception as e:  # 再検出自体が壊れたら通さない
        out["V4"] = {"pass": False, "code": "V_REDETECT_TRACE", "metrics": {"exception": f"{type(e).__name__}: {e}"}}
    out["V5"] = v5_boundary_ink(orig_arr, geom, tmpl, cfg)
    if orig_path is not None and orig_before is not None:
        out["V6"] = v6_original(orig_path, orig_before)
    else:
        out["V6"] = {"pass": False, "code": "X_ORIGINAL_CHANGED", "metrics": {"skipped": True}}
    out["V7"] = v7_ink_outside(orig_arr, geom, tmpl, cfg)
    out["pass"] = all(out[k]["pass"] for k in VERDICTS)
    out["mask_pixels"] = int(mask.sum())
    return out


def failed_reasons(vr: dict) -> list:
    return [{"code": vr[k]["code"], "stage": "S8", "metric": k, "value": False, "threshold": True}
            for k in VERDICTS if not vr[k]["pass"]]
