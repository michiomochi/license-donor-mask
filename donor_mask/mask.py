"""S7 マスク矩形の作成（位置合わせ済みテンプレート座標で作る）。

R（T ∪ F ∪ A）に余白を足し、保護領域を避け、境界インクで辺ごとに広げる。決められなければ review。
元画像への写像と塗りは pipeline で行う。検証（verify.py）はこのモジュールを import しない。
"""
from __future__ import annotations

from . import ink as inkmod
from .geom import rect_area, rect_inter


def base_pads(R, cfg):
    """(left, top, right, bottom) の基本余白。各辺 pad_cap_px が上限。"""
    m = cfg.mask
    px = min(max(m.pad_x_min_px, m.pad_x_ratio * (R[2] - R[0])), m.pad_cap_px)
    py = min(max(m.pad_y_min_px, m.pad_y_ratio * (R[3] - R[1])), m.pad_cap_px)
    return [px, py, px, py]


def padded(R, pads):
    return (R[0] - pads[0], R[1] - pads[1], R[2] + pads[2], R[3] + pads[3])


def clip(r, cfg):
    return (max(0.0, r[0]), max(0.0, r[1]), min(float(cfg.canvas.width), r[2]), min(float(cfg.canvas.height), r[3]))


def shrink_for_protected(R, pads, protected, cfg):
    """保護領域と重なる辺の余白だけを縮める。縮めた後も最低 protected_min_gap_px 残らなければ None。"""
    m = cfg.mask
    pads = list(pads)
    log = []
    clear = float(m.protected_clear_px)
    for p in protected:
        if rect_inter(padded(R, pads), p) <= 0:
            continue
        if rect_inter(R, p) > 0:
            return None, log
        opts = []  # (側index, R から保護領域までの隙間)
        if p[2] <= R[0]:
            opts.append((0, R[0] - p[2]))
        if p[0] >= R[2]:
            opts.append((2, p[0] - R[2]))
        if p[3] <= R[1]:
            opts.append((1, R[1] - p[3]))
        if p[1] >= R[3]:
            opts.append((3, p[1] - R[3]))
        if not opts:
            return None, log
        side, gap = max(opts, key=lambda o: (o[1], -o[0]))
        newpad = min(pads[side], gap - clear)
        log.append({"protected": list(p), "side": "ltrb"[side], "pad_before": pads[side], "pad_after": newpad})
        if newpad < m.protected_min_gap_px:
            return None, log
        pads[side] = newpad
    return pads, log


def build_mask_rect(aligned_bgr, tmpl, cues, cfg):
    """(rect|None, metrics, reasons)。rect はテンプレート座標の軸平行矩形。"""
    m, k = cfg.mask, cfg.ink
    R = cues["R"]
    prot = [tuple(float(v) for v in p) for p in tmpl.meta["protected_rects"]]
    met, reasons = {"R": list(R)}, []

    def fail(code, metric, value, thr):
        reasons.append({"code": code, "metric": metric, "value": value, "threshold": thr})
        return None, met, reasons

    pads = base_pads(R, cfg)
    met["pads_base"] = list(pads)
    pads, plog = shrink_for_protected(R, pads, prot, cfg)
    met["protected_shrink"] = plog
    if pads is None:
        return fail("M_PROTECTED_CONFLICT", "pad_after_shrink", plog[-1]["pad_after"] if plog else 0,
                    m.protected_min_gap_px)
    exp_area = float(tmpl.meta["expected_mask_area"])

    ink = inkmod.ink_map(aligned_bgr, tmpl.print_map, cfg)
    rect = clip(padded(R, pads), cfg)
    steps = []
    for it in range(int(k.max_expand_times) + 1):
        bad, im = inkmod.check_boundary_ink(ink, rect, cfg)
        steps.append({"rect": list(rect), "bad_sides": sorted(bad), "ink": im})
        if not bad:
            break
        if it == int(k.max_expand_times):
            met["ink_steps"] = steps
            return fail("M_INK_OVERFLOW", "bad_sides_after_expand", sorted(bad), [])
        for s in bad:
            i = "ltrb".index(s)
            pads[i] += float(k.expand_px)
        new = clip(padded(R, pads), cfg)
        if any(rect_inter(new, p) > 0 for p in prot):
            met["ink_steps"] = steps
            return fail("M_INK_OVERFLOW", "expand_hits_protected", sorted(bad), [])
        rect = new
    met["ink_steps"] = steps
    met["pads_final"] = list(pads)
    ratio = rect_area(rect) / exp_area
    met["area_ratio"] = ratio
    if not (m.area_ratio_min <= ratio <= m.area_ratio_max):
        return fail("M_AREA_OUT_OF_RANGE", "area_ratio", ratio, [m.area_ratio_min, m.area_ratio_max])
    return rect, met, reasons
