"""run のログ（out/logs/<job_id>.json）の集計。画像・パス・job_id・ファイル名は一切出さない（件数と分布だけ）。

用途: 本番の環境（実写真のある場所）で人が実行し、しきい値の基準を決める材料にする。
実行時に AI は使わない・特定の画像に合わせた調整もしない（集計は固定の定義だけ）。

集めるもの:
  * status（ok/review/error）の件数、mode（run/diagnostic）の件数
  * 理由コード別件数（rec.reasons の code を 1 件につき 1 回数える）と一次理由（primary_reason）別件数
  * 分布（status 別・全体）: S3 の best ZNCC / S4 の ECC 相関 / カード長辺 px（S5。S1 後のカード長辺）/ 入力長辺 px（S0）
    — 要約統計（min/p5/p25/p50/p75/p95/max）と固定幅のヒストグラム
ログに無い値（その段まで進まなかった件）は数に入れない。各分布の n に、値のあった件数を出す。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

STATUSES = ("ok", "review", "error")
QUANTILES = (("min", 0), ("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95), ("max", 100))

# 分布の名前 → (ログ内の位置, ヒストグラムの区切り)。区切りは固定（データから決めない）。
# 区切りの外（下側・上側）は "<lo" と ">=hi" に数える。
_ZNCC_EDGES = [round(0.1 * i, 1) for i in range(0, 11)]
_RES_EDGES = [0, 200, 400, 600, 800, 1200, 1600, 2400, 3200, 4800]
METRICS = {
    "zncc": (("stages", "S3", "best", "zncc"), _ZNCC_EDGES),
    "ecc_correlation": (("stages", "S4", "correlation"), _ZNCC_EDGES),
    "card_long_edge_px": (("stages", "S5", "card_long_edge_px"), _RES_EDGES),
    "input_long_edge_px": (("stages", "S0", "long_edge_px"), _RES_EDGES),
}


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _num(v):
    """有限の数値だけ通す（bool・文字列・None・NaN/inf は捨てる）。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if np.isfinite(v) else None


def is_job_log(rec) -> bool:
    """1件ごとのログか。batch_*.json（jobs を持つ集計）や別形式の JSON は除く。"""
    return (isinstance(rec, dict) and rec.get("status") in STATUSES and isinstance(rec.get("reasons"), list)
            and "jobs" not in rec)


def find_log_files(p: Path) -> list[Path]:
    """p がログのディレクトリ、または run の出力先（その下の logs/）。サブディレクトリは掘らない。ソート済み。"""
    p = Path(p)
    d = p / "logs" if (p / "logs").is_dir() else p
    return sorted((f for f in d.glob("*.json") if f.is_file() and not f.name.startswith("batch_")), key=lambda f: f.name)


def load_logs(files) -> tuple[list[dict], int]:
    """(1件ごとのログのリスト, 読めない/形式違いで除いた数)。"""
    recs, skipped = [], 0
    for f in files:
        try:
            rec = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped += 1
            continue
        if is_job_log(rec):
            recs.append(rec)
        else:
            skipped += 1
    return recs, skipped


def summarize(values) -> dict:
    v = np.asarray([x for x in values if x is not None], np.float64)
    if v.size == 0:
        return {"n": 0}
    out = {"n": int(v.size)}
    for name, q in QUANTILES:
        out[name] = round(float(np.percentile(v, q)), 4)
    return out


def histogram(values, edges) -> dict:
    """固定の区切りでの件数。キーは "<lo" / "lo-hi" / ">=hi"。"""
    v = np.asarray([x for x in values if x is not None], np.float64)
    h = {f"<{edges[0]:g}": int((v < edges[0]).sum())}
    for lo, hi in zip(edges[:-1], edges[1:]):
        h[f"{lo:g}-{hi:g}"] = int(((v >= lo) & (v < hi)).sum())
    h[f">={edges[-1]:g}"] = int((v >= edges[-1]).sum())
    return h


def aggregate(recs: list[dict], mode: str = "any") -> dict:
    """集計結果（JSON 化できる dict）。mode は any / run / diagnostic。"""
    if mode != "any":
        recs = [r for r in recs if r.get("mode") == mode]
    status = Counter(r["status"] for r in recs)
    modes = Counter(str(r.get("mode")) for r in recs)
    reasons, primary = Counter(), Counter()
    for r in recs:
        for code in dict.fromkeys(x.get("code") for x in r["reasons"] if isinstance(x, dict) and x.get("code")):
            reasons[code] += 1
        primary[r.get("primary_reason") or ("(none)" if r["status"] == "ok" else "(unrecorded)")] += 1
    metrics = {}
    for name, (path, edges) in METRICS.items():
        by = {"all": [_num(_dig(r, path)) for r in recs]}
        for s in STATUSES:
            by[s] = [_num(_dig(r, path)) for r in recs if r["status"] == s]
        metrics[name] = {s: {**summarize(vs), "hist": histogram(vs, edges)} for s, vs in by.items() if s == "all" or any(
            x is not None for x in vs)}
    n = len(recs)
    return {
        "n_logs": n,
        "mode": dict(sorted(modes.items())),
        "status": {s: status.get(s, 0) for s in STATUSES},
        "ok_rate": round(status.get("ok", 0) / n, 4) if n else None,
        "reason_counts": dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        "primary_reason_counts": dict(sorted(primary.items(), key=lambda kv: (-kv[1], kv[0]))),
        "metrics": metrics,
    }


def format_text(agg: dict, skipped: int = 0) -> str:
    L = [f"logs: {agg['n_logs']} 件（読めない/形式違いで除外: {skipped}）  mode: {agg['mode']}",
         f"status: {agg['status']}  ok 率: {agg['ok_rate']}"]
    if agg["mode"].get("diagnostic"):
        L.append("[注意] diagnostic のログは status が常に review。ok/review の比較は run のログだけで行う（--mode run）")
    L.append("")
    L.append("理由コード別（1件に複数ありうる。同一コードは1件1回）:")
    L += [f"  {c:<28}{n:>6}" for c, n in agg["reason_counts"].items()] or ["  (なし)"]
    L.append("一次理由別:")
    L += [f"  {c:<28}{n:>6}" for c, n in agg["primary_reason_counts"].items()] or ["  (なし)"]
    for name, by in agg["metrics"].items():
        L += ["", f"{name}:"]
        for s, d in by.items():
            if d["n"] == 0:
                L.append(f"  [{s}] n=0")
                continue
            L.append(f"  [{s}] n={d['n']}  " + "  ".join(f"{k}={d[k]:g}" for k, _ in QUANTILES))
            L.append("       hist: " + "  ".join(f"{k}:{v}" for k, v in d["hist"].items() if v))
    return "\n".join(L)
