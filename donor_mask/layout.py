"""罫線マップ・レイアウト署名・ZNCC（S2 の部品）。検出側・検証側で共通に使う。"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def dark_map(img, cfg) -> np.ndarray:
    """暗い画素（印刷インク候補）: adaptiveThreshold(MEAN_C, BINARY_INV)。"""
    s = cfg.signature
    return cv2.adaptiveThreshold(to_gray(img), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                 int(s.block_size), float(s.c))


def ruling_from_dark(dark, cfg) -> np.ndarray:
    """横長(41x1)・縦長(1x25)カーネルでそれぞれ OPEN して OR。文字は消え、罫線だけが残る。"""
    s = cfg.signature
    h = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(s.kernel_h), 1)))
    v = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(s.kernel_v))))
    return cv2.bitwise_or(h, v)


def ruling_map(img, cfg) -> np.ndarray:
    return ruling_from_dark(dark_map(img, cfg), cfg)


def zero_rects(m: np.ndarray, rects) -> np.ndarray:
    out = m.copy()
    for (x0, y0, x1, y1) in rects or []:
        out[max(0, int(round(y0))):int(round(y1)), max(0, int(round(x0))):int(round(x1))] = 0
    return out


def signature(ruling: np.ndarray, ignore_rects, cfg) -> np.ndarray:
    """ignore_rects を 0 にして 101x64 へ INTER_AREA 縮小した float32（必要なら弱くぼかす）。"""
    s = cfg.signature
    m = zero_rects(ruling, ignore_rects)
    sig = cv2.resize(m, (int(s.out_width), int(s.out_height)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if float(s.blur_sigma) > 0:
        sig = cv2.GaussianBlur(sig, (0, 0), float(s.blur_sigma))
    return sig


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    """零平均正規化相互相関。分散0なら 0（一致とみなさない）。"""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0
