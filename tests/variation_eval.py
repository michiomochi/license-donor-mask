"""【評価】variation_gen で作った撮影バリエーションを本番と同じ経路（pipeline.process_file）に通し、
ok 率と誤検出（ok になったのに塗りが正しくない）を測る。

誤検出（false_ok）の定義 — ok になった件だけが対象。1 つでも該当すれば誤検出:
  misplaced        塗り位置ずれ: GT の欄（1px 侵食）に、塗られていない画素がある（包含率 < 1）
  outside_card     カード以外を塗る: 原本から変わった画素が、カード四角形（3px 膨張）の外にある
  protected        保護領域（2px 縮小）の画素が変わった
  overpaint        塗りすぎ: 変わった画素数が GT の欄の面積の 1.6 倍を超える
  not_fill_colour  塗り色以外の変更がある（欄外の差分）

ok でない件は review（または error）として理由コードを数える（取りこぼしの側。誤検出ではない）。
出力は件数と率だけ。画像・ファイル名・パスは出さない。一時ファイルは評価後に消す。
"""
from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import fixtures_gen as fx
import variation_gen as vg
from donor_mask import imgio, pipeline

FALSE_OK_KINDS = ("misplaced", "outside_card", "protected", "overpaint", "not_fill_colour")
OVERPAINT_MAX = 1.6
CARD_DILATE_PX = 3


PROTECTED_TOL_PX = 2.0  # 既存の不変条件（conftest.assert_ok_invariants）と同じ: 保護領域を 2px 縮めた内側に塗りが入れば違反


def protected_depth_px(fixture: fx.Fixture, changed: np.ndarray) -> float:
    """変わった画素が保護領域（GT）の内側へ最大で何 px 入ったか（入っていなければ 0）。"""
    pm = np.zeros(changed.shape, np.uint8)
    for p in fixture.gt_protected:
        cv2.fillPoly(pm, [np.round(p).astype(np.int32)], 1)
    if not pm.any() or not (changed & pm.astype(bool)).any():
        return 0.0
    return float(cv2.distanceTransform(pm, cv2.DIST_L2, 3)[changed].max())


def analyze_ok(fixture: fx.Fixture, rec: dict, fill=(0, 0, 0)) -> dict:
    """ok の 1 件を GT と突き合わせる: {"kinds": [...], "protected_depth_px": float}。"""
    kinds, depth = _analyze(fixture, rec, fill)
    return {"kinds": kinds, "protected_depth_px": depth}


def false_ok_kinds(fixture: fx.Fixture, rec: dict, fill=(0, 0, 0)) -> list[str]:
    """rec が ok のときの誤検出の種類（空なら正しい ok）。出力 PNG を読み戻して GT と突き合わせる。"""
    return _analyze(fixture, rec, fill)[0]


def _analyze(fixture, rec, fill):
    out = cv2.imdecode(np.fromfile(rec["outputs"]["masked"], np.uint8), cv2.IMREAD_COLOR)
    orig, _ = imgio.decode(fixture.data, 10**9)
    if out is None or out.shape != orig.shape:
        return ["not_fill_colour"], 0.0
    kinds = []
    filled = np.all(out == np.array(fill, np.uint8), axis=2)
    changed = np.any(out != orig, axis=2)
    fm = fx.field_mask(out.shape, fixture, erode=1)
    if fm.sum() == 0 or float((fm & filled).sum()) / float(fm.sum()) < 1.0:
        kinds.append("misplaced")
    cardm = np.zeros(out.shape[:2], np.uint8)
    cv2.fillPoly(cardm, [np.round(fixture.quad).astype(np.int32)], 1)
    k = 2 * CARD_DILATE_PX + 1
    cardm = cv2.dilate(cardm, np.ones((k, k), np.uint8)).astype(bool)
    if (changed & ~cardm).any():
        kinds.append("outside_card")
    depth = protected_depth_px(fixture, changed)
    if depth > PROTECTED_TOL_PX:
        kinds.append("protected")
    gt_full = fx.field_mask(out.shape, fixture, erode=0)
    if float(changed.sum()) / max(float(gt_full.sum()), 1.0) > OVERPAINT_MAX:
        kinds.append("overpaint")
    if (changed & ~filled).any():
        kinds.append("not_fill_colour")
    return kinds, depth


def evaluate_case(sample: vg.Sample, case: dict, cfg, tmpls, workdir: Path) -> dict:
    """1 ケース → {status, reasons(コードだけ), false_ok(種類のリスト)}。画像は workdir に書いて直後に消す。"""
    f = vg.make_variation(sample, case)
    templates, tsha = tmpls
    d = Path(workdir)
    inp = d / ("in" + f.ext)
    inp.write_bytes(f.data)
    out = d / "out"
    try:
        rec, _halt = pipeline.process_file(inp, out, cfg, templates, tsha)
        an = analyze_ok(f, rec) if rec["status"] == "ok" else {"kinds": [], "protected_depth_px": 0.0}
        return {"status": rec["status"], "reasons": list(dict.fromkeys(r["code"] for r in rec["reasons"])),
                "primary": rec.get("primary_reason"), "false_ok": an["kinds"], "protected_depth_px": an["protected_depth_px"]}
    finally:
        inp.unlink(missing_ok=True)
        shutil.rmtree(out, ignore_errors=True)


def evaluate_grid(samples: list[vg.Sample], grid: dict[str, list[dict]], cfg, tmpls, axes=None, progress=None) -> dict:
    """軸ごとの集計。samples が複数なら全見本×全ケース。見本ごとの内訳は出さない（個別の画像を特定させない）。"""
    res = {}
    tmp = Path(tempfile.mkdtemp(prefix="donor_mask_eval_"))
    try:
        for axis, cases in grid.items():
            if axes and axis not in axes:
                continue
            a = {"n": 0, "ok": 0, "review": 0, "error": 0, "false_ok": 0, "false_ok_kinds": Counter(), "reasons": Counter(),
                 "ok_cases": [], "false_ok_cases": [], "max_protected_depth_px": 0.0}
            for si, s in enumerate(samples):
                for ci, case in enumerate(cases):
                    r = evaluate_case(s, case, cfg, tmpls, tmp)
                    a["n"] += 1
                    a[r["status"]] += 1
                    if r["status"] == "ok":
                        a["ok_cases"].append((si, ci))
                        a["max_protected_depth_px"] = max(a["max_protected_depth_px"], r["protected_depth_px"])
                    if r["false_ok"]:
                        a["false_ok"] += 1
                        a["false_ok_kinds"].update(r["false_ok"])
                        a["false_ok_cases"].append((si, ci, r["false_ok"]))
                    if r["status"] != "ok":
                        a["reasons"][r["primary"] or "(none)"] += 1
                    if progress:
                        progress(axis, a["n"])
            res[axis] = a
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def summarize(res: dict) -> dict:
    """JSON 化できる集計（ケースの添字は含めるが画像の情報は含まない）。"""
    out = {"axes": {}, "total": {"n": 0, "ok": 0, "review": 0, "error": 0, "false_ok": 0}}
    for axis, a in res.items():
        out["axes"][axis] = {"n": a["n"], "ok": a["ok"], "review": a["review"], "error": a["error"],
                             "ok_rate": round(a["ok"] / a["n"], 4) if a["n"] else None, "false_ok": a["false_ok"],
                             "false_ok_kinds": dict(a["false_ok_kinds"]),
                             "max_protected_depth_px": round(a["max_protected_depth_px"], 2),
                             "review_primary_reasons": dict(sorted(a["reasons"].items(), key=lambda kv: (-kv[1], kv[0]))),
                             "false_ok_cases": [[si, ci, k] for si, ci, k in a["false_ok_cases"]]}
        for k in ("n", "ok", "review", "error", "false_ok"):
            out["total"][k] += a[k]
    t = out["total"]
    t["max_protected_depth_px"] = max((a["max_protected_depth_px"] for a in out["axes"].values()), default=0.0)
    t["ok_rate"] = round(t["ok"] / t["n"], 4) if t["n"] else None
    t["false_ok_rate_of_ok"] = round(t["false_ok"] / t["ok"], 4) if t["ok"] else None
    return out


def format_text(summary: dict) -> str:
    L = [f"{'axis':<12}{'n':>5}{'ok':>5}{'review':>8}{'error':>7}{'ok率':>8}{'誤検出':>8}  review の一次理由 / 誤検出の種類"]
    for axis, a in summary["axes"].items():
        why = ", ".join(f"{k}:{v}" for k, v in a["review_primary_reasons"].items())
        fo = ", ".join(f"{k}:{v}" for k, v in a["false_ok_kinds"].items())
        L.append(f"{axis:<12}{a['n']:>5}{a['ok']:>5}{a['review']:>8}{a['error']:>7}{a['ok_rate']:>8.3f}{a['false_ok']:>8}  {why}" + (f"  [誤検出: {fo}]" if fo else ""))
    t = summary["total"]
    L.append(f"{'TOTAL':<12}{t['n']:>5}{t['ok']:>5}{t['review']:>8}{t['error']:>7}{t['ok_rate']:>8.3f}{t['false_ok']:>8}")
    L.append(f"ok のうち保護領域へ入った最大の深さ: {t['max_protected_depth_px']:.2f}px（誤検出にするのは {PROTECTED_TOL_PX:g}px 超）")
    return "\n".join(L)
