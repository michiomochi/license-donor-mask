"""幾何の小道具。検出側・検証側の両方から使う（マスク処理には依存しない）。"""
from __future__ import annotations

import cv2
import numpy as np


def order_quad(pts) -> np.ndarray:
    """4点を 左上,右上,右下,左下（画像座標で時計回り）に並べる。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang, kind="stable")]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def edge_lengths(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)  # 上,右,下,左


def interior_angles(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = []
    for i in range(4):
        a, b, c = q[(i - 1) % 4], q[i], q[(i + 1) % 4]
        u, v = a - b, c - b
        cosv = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        out.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    return np.array(out)


def quad_area(q) -> float:
    return float(abs(cv2.contourArea(np.asarray(q, dtype=np.float32).reshape(-1, 1, 2))))


def quad_diag(q) -> float:
    q = np.asarray(q, dtype=np.float64)
    return float(max(np.linalg.norm(q[2] - q[0]), np.linalg.norm(q[3] - q[1])))


def quad_aspect(q) -> float:
    """長辺平均 / 短辺平均（90度回転に依存しない見かけのアスペクト）。"""
    e = edge_lengths(q)
    a, b = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
    return float(max(a, b) / max(min(a, b), 1e-9))


def quad_dist(q1, q2) -> float:
    """同じ向きの4角の対応点距離の最大値（巡回4通りの最小）。"""
    q1, q2 = np.asarray(q1, np.float64), np.asarray(q2, np.float64)
    return float(min(np.linalg.norm(np.roll(q2, -k, axis=0) - q1, axis=1).max() for k in range(4)))


def quad_iou(q1, q2) -> float:
    a = np.asarray(q1, np.float32).reshape(-1, 1, 2)
    b = np.asarray(q2, np.float32).reshape(-1, 1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    ua = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


def hypothesis_quads(q) -> list:
    """4回転 x 鏡像 = 8仮説（固定順序）。[(rot, mirrored, quad)]。長辺を横にする前処理はしない。"""
    q = order_quad(q)
    out = []
    for mirrored in (False, True):
        for k in range(4):
            r = np.roll(q, -k, axis=0)
            if mirrored:
                r = r[[1, 0, 3, 2]]  # 左右反転
            out.append((k * 90, mirrored, r))
    return out


def canon_matrix(quad, w: int, h: int) -> np.ndarray:
    """正規化座標(w x h) -> 元画像座標 の射影行列。"""
    canon = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(canon, np.asarray(quad, np.float32))


def warp_to_frame(img, M, w: int, h: int):
    """M: 目的座標系 -> 元画像座標。目的座標系の画像を作る。"""
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def apply_h(M, pts) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(M, np.float64)).reshape(-1, 2)


def affine3(A) -> np.ndarray:
    A = np.asarray(A, np.float64)
    return np.vstack([A, [0, 0, 1]]) if A.shape == (2, 3) else A


# ---- 矩形 (x0, y0, x1, y1)
def rect_area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def rect_inter(a, b) -> float:
    return rect_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def rect_clip(a, b) -> tuple:
    """a と b の共通部分の矩形（重なりが無ければ面積0の矩形）。"""
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def rect_iou(a, b) -> float:
    i = rect_inter(a, b)
    u = rect_area(a) + rect_area(b) - i
    return float(i / u) if u > 0 else 0.0


def rect_union_bbox(rs) -> tuple:
    return (min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs))


def rect_poly(r) -> np.ndarray:
    return np.array([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], dtype=np.float64)


def rect_inside(inner, outer, tol=1e-6) -> bool:
    return inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and \
        inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol


def polygon_mask(shape, poly, grow: int) -> np.ndarray:
    """多角形を塗ったブール mask。grow px だけ外へ広げる（丸め誤差の吸収）。"""
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    if grow > 0:
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return m.astype(bool)
