"""verify._protected_strokes: マスクから続いて保護領域へはみ出した長い一画だけを拾う。"""
from types import SimpleNamespace

import cv2
import numpy as np

from donor_mask.verify import _protected_strokes

W, H = 400, 300
POLY = np.array([[100, 200], [300, 200], [300, 260], [100, 260]], float)
TMPL = SimpleNamespace(meta={"protected_rects": [[0, 0, 400, 200]]})


def _k(n=40):
    return SimpleNamespace(protected_stroke_px=n, ring_px=6)


def _ink(*lines):
    im = np.zeros((H, W), np.uint8)
    for a, b in lines:
        cv2.line(im, a, b, 255, 3)
    return im


def test_long_stroke_from_mask_is_found():
    assert _protected_strokes(_ink(((200, 205), (200, 80))), POLY, TMPL, _k(), W, H)


def test_short_text_near_mask_is_not_counted():
    assert not _protected_strokes(_ink(((120, 190), (180, 190))), POLY, TMPL, _k(), W, H)


def test_far_text_not_touching_mask_is_not_counted():
    assert not _protected_strokes(_ink(((50, 20), (50, 120))), POLY, TMPL, _k(), W, H)


def test_disabled_by_zero():
    assert not _protected_strokes(_ink(((200, 205), (200, 80))), POLY, TMPL, _k(0), W, H)
