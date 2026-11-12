"""境界インク（手書きのはみ出し）の検出。S7 と V5 が共通に使う検出側の部品。

位置合わせ済みテンプレート座標系で、テンプレートの印刷物（print.png）を除いた暗画素＝インクを数える。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import layout



def ink_map(aligned_bgr, print_map, cfg) -> np.ndarray:
    """print_map が None のときは「印刷物マップ無し」＝暗画素を全部インクとして扱う（保守側）。"""
    k = cfg.ink
    dark = layout.dark_map(aligned_bgr, cfg)
    if print_map is None:
        ink = dark.copy()
    else:
        pd = 2 * int(k.print_dilate_px) + 1
        printed = cv2.dilate(print_map, np.ones((pd, pd), np.uint8))
        ink = np.where((dark > 0) & (printed == 0), 255, 0).astype(np.uint8)
    b = int(k.border_ignore_px)  # カード縁（背景が写り込む）は印刷物でも手書きでもないので数えない
    if b > 0:
        ink[:b, :] = 0
        ink[-b:, :] = 0
        ink[:, :b] = 0
        ink[:, -b:] = 0
    o = int(k.open_px)
    if o > 1:
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((o, o), np.uint8))
    return ink


def _irect(rect, shape):
    H, W = shape[:2]
    x0, y0 = max(0, math.floor(rect[0])), max(0, math.floor(rect[1]))
    x1, y1 = min(W, math.ceil(rect[2])), min(H, math.ceil(rect[3]))
    return x0, y0, x1, y1


def strips(rect, ring: int, shape) -> dict:
    """矩形の外側 ring px の4辺ストリップ（x0,y0,x1,y1）。角は上下に含める。"""
    H, W = shape[:2]
    x0, y0, x1, y1 = _irect(rect, shape)
    c = lambda v, hi: max(0, min(hi, v))  # noqa: E731
    return {
        "t": (c(x0 - ring, W), c(y0 - ring, H), c(x1 + ring, W), y0),
        "b": (c(x0 - ring, W), y1, c(x1 + ring, W), c(y1 + ring, H)),
        "l": (c(x0 - ring, W), y0, x0, y1),
        "r": (x1, y0, c(x1 + ring, W), y1),
    }


def check_boundary_ink(ink: np.ndarray, rect, cfg):
    """(bad_sides:set, metrics)。辺ごとのリング内インク比と、境界をまたぐ長さ10px以上の連結成分を見る。"""
    k = cfg.ink
    ring = int(k.ring_px)
    st = strips(rect, ring, ink.shape)
    bad, met = set(), {"ratio": {}, "crossing": []}
    for s, (a, b, c, d) in st.items():
        area = max(1, (c - a) * (d - b))
        r = float(np.count_nonzero(ink[b:d, a:c])) / area
        met["ratio"][s] = r
        if r > k.ink_ratio_max:
            bad.add(s)
    x0, y0, x1, y1 = _irect(rect, ink.shape)
    H, W = ink.shape
    ex0, ey0, ex1, ey1 = max(0, x0 - ring), max(0, y0 - ring), min(W, x1 + ring), min(H, y1 + ring)
    sub = ink[ey0:ey1, ex0:ex1]
    inside = np.zeros(sub.shape, bool)
    inside[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = True
    n, lab, stats, _ = cv2.connectedComponentsWithStats((sub > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        comp = lab == i
        if not (comp & inside).any() or not (comp & ~inside).any():
            continue
        if max(stats[i][2], stats[i][3]) < k.cross_min_len_px:
            continue
        outside_pts = comp & ~inside
        for s, (a, b, c, d) in st.items():
            sa, sb, sc, sd = a - ex0, b - ey0, c - ex0, d - ey0
            if sc > sa and sd > sb and outside_pts[sb:sd, sa:sc].any():
                bad.add(s)
                met["crossing"].append(s)
    # 塗りが厳密でなくてよい辺（tolerate_l/t/r/b = 1。例: t は保護領域に接する上辺）は、インク比の超過だけ失敗にしない。
    # 境界をまたぐ線（crossing）はペンの一画の可能性があるので、許容辺でも失敗に残す。
    bad -= {sd for sd in "ltrb" if int(getattr(k, "tolerate_" + sd)) == 1 and sd not in met["crossing"]}
    return bad, met
