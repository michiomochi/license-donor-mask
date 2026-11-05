"""S5 品質ゲート（欄の探索窓の中で測る）。"""
from __future__ import annotations

import cv2
import numpy as np

from .template import window_rect


def check_quality(aligned_bgr, template, card_long_edge_px: float, cfg):
    """(metrics, reasons)。窓は位置合わせ済み画像（テンプレート座標）上の探索窓。"""
    q = cfg.quality
    x0, y0, x1, y1 = [int(round(v)) for v in window_rect(template, cfg)]
    win = aligned_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
    glare = float(np.mean(np.all(win >= q.glare_level, axis=2)))
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p5)
    met = {"glare_ratio": glare, "laplacian_var": lap, "contrast_p95_p5": contrast,
           "card_long_edge_px": float(card_long_edge_px)}
    reasons = []
    if glare > q.glare_max_ratio:
        reasons.append({"code": "Q_GLARE", "metric": "glare_ratio", "value": glare, "threshold": q.glare_max_ratio})
    if lap < q.blur_min_laplacian_var:
        reasons.append({"code": "Q_BLUR", "metric": "laplacian_var", "value": lap,
                        "threshold": q.blur_min_laplacian_var})
    if contrast < q.contrast_min_p95_p5:
        reasons.append({"code": "Q_CONTRAST", "metric": "contrast_p95_p5", "value": contrast,
                        "threshold": q.contrast_min_p95_p5})
    return met, reasons
