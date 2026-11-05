"""CLI: python -m donor_mask {run,rectify,verify,template-build}。calibrate / make-fixtures は v1 では未実装。

終了コード: 0=全件 ok / 3=review（または件単位の error）を含む / 1=設定・テンプレート不正や I/O の致命的エラー
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import build, card, imgio, orient, pipeline, verify
from .config import DEFAULT_CONFIG, DEFAULT_TEMPLATES, ConfigError, load_config
from .geom import hypothesis_quads
from .template import load_templates

DEFERRED = ("calibrate", "make-fixtures")


def collect_inputs(p: Path):
    """ファイル1つ、またはディレクトリ直下の jpg/jpeg/png（サブディレクトリは掘らない。ソート済み）。"""
    if p.is_dir():
        return sorted((f for f in p.iterdir() if f.is_file() and f.suffix.lower() in imgio.EXTS), key=lambda f: f.name)
    return [p]


def check_dirs(src: Path, out: Path) -> str | None:
    """入力と出力が同じ、または出力が入力の配下なら理由文字列（=起動エラー）。"""
    in_dir = (src if src.is_dir() else src.parent).resolve()
    o = out.resolve()
    if o == in_dir or in_dir in o.parents:
        return f"出力先が入力ディレクトリと同じ、または入力の配下です: out={o} in={in_dir}"
    return None


def draw_grid(img, step=0.05):
    """5% 刻みのグリッドと % ラベルを焼き込む（10% 刻みは太線）。校正用。"""
    vis = img.copy()
    h, w = vis.shape[:2]
    fs = max(0.35, max(h, w) / 1000 * 0.4)
    n = int(round(1 / step))
    for i in range(n + 1):
        pct = int(round(i * step * 100))
        major = pct % 10 == 0
        color, th = ((255, 0, 255), 2) if major else ((255, 160, 0), 1)
        x, y = min(w - 1, int(round(i * step * w))), min(h - 1, int(round(i * step * h)))
        cv2.line(vis, (x, 0), (x, h - 1), color, th, cv2.LINE_AA)
        cv2.line(vis, (0, y), (w - 1, y), color, th, cv2.LINE_AA)
        for (px, py) in ((x + 2, int(14 * fs * 2)), (2, y - 2 if y > 12 else y + int(14 * fs * 2))):
            cv2.putText(vis, str(pct), (px, py), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(vis, str(pct), (px, py), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def cmd_run(a) -> int:
    cfg = load_config(a.config)
    templates, tsha = load_templates(a.templates, cfg)
    src, out = Path(a.inp).expanduser(), Path(a.out).expanduser()
    if not src.exists():
        print(f"[error] 入力が存在しません: {src}", file=sys.stderr)
        return 1
    bad = check_dirs(src, out)
    if bad:
        print(f"[error] {bad}", file=sys.stderr)
        return 1
    files = collect_inputs(src)
    if not files:
        print(f"[error] 対象画像(jpg/jpeg/png)がありません: {src}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    results, halted = [], False
    for f in files:
        rec, halt = pipeline.process_file(f, out, cfg, templates, tsha, fill=a.fill, shadow=a.shadow)
        results.append(rec)
        codes = ",".join(dict.fromkeys(r["code"] for r in rec["reasons"]))
        print(f"[{rec['status']}] {f.name} -> job {rec['job_id']}" + (f"  reasons: {codes}" if codes else ""))
        if halt:
            why = ",".join(dict.fromkeys(r["code"] for r in rec["reasons"]))
            print(f"[FATAL] 原本が処理中に変化、または ok/ の後始末に失敗しました（{why}）。バッチを停止します",
                  file=sys.stderr)
            halted = True
            break
    shas = sorted(r["input"].get("sha256", r["job_id"]) for r in results)
    set8 = hashlib.sha256("\n".join(shas).encode()).hexdigest()[:8]
    summary = {"schema_version": 1, "config_sha256": cfg.sha256, "templates_sha256": tsha, "fill": a.fill,
               "shadow": a.shadow, "halted": halted,
               "counts": {s: sum(1 for r in results if r["status"] == s) for s in ("ok", "review", "error")},
               "jobs": [{"job_id": r["job_id"], "status": r["status"],
                         "reasons": list(dict.fromkeys(x["code"] for x in r["reasons"]))} for r in results]}
    bp = out / "logs" / f"batch_{cfg.sha256[:8]}_{set8}.json"
    bp.parent.mkdir(parents=True, exist_ok=True)
    if bp.exists():
        print(f"[warn] 既存のバッチログは上書きしません: {bp}", file=sys.stderr)
    else:
        bp.write_text(json.dumps(pipeline.clean(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {summary['counts']}  logs: {bp.parent}")
    if halted:
        return 1
    return 0 if all(r["status"] == "ok" for r in results) else 3


def cmd_rectify(a) -> int:
    """校正用: 各カード候補の「最良の非鏡像仮説」を 1011x638 に直し、5%グリッドを焼き込む（判定はしない）。"""
    cfg = load_config(a.config)
    templates, _ = load_templates(a.templates, cfg)
    src, out = Path(a.inp).expanduser(), Path(a.out).expanduser()
    bad = check_dirs(src, out)
    if bad:
        print(f"[error] {bad}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for f in collect_inputs(src):
        data = f.read_bytes()
        job = imgio.sha256_bytes(data)[:16]
        try:
            img, _info = imgio.decode(data, int(cfg.input.max_pixels))
        except imgio.InputError as e:
            print(f"[error] {f.name}: {e}", file=sys.stderr)
            continue
        pg = out / f"{job}__orig_grid.jpg"
        pg.write_bytes(imgio.encode_jpeg(draw_grid(img), 90))
        print(f"[grid] 元画像のグリッド: {pg}")
        cands = card.find_candidates(img, cfg)
        print(f"[rectify] {f.name}: card candidates={len(cands['candidates'])} rejected={cands['rejected']}")
        for i, c in enumerate(cands["candidates"]):
            hyps = [h for h in orient.score_hypotheses(img, [c], templates, cfg) if not h["mirror"] and h["kind"] == "positive"]
            best = max(hyps, key=lambda h: h["zncc"])
            q = hypothesis_quads(c["quad"])[best["rot"] // 90][2]
            norm, _ = orient.normalize(img, q, cfg)
            p1, p2 = out / f"{job}__rect{i}.png", out / f"{job}__rect{i}_grid.jpg"
            p1.write_bytes(imgio.encode_png(norm, int(cfg.output.png_compression)))
            p2.write_bytes(imgio.encode_jpeg(draw_grid(norm), 90))
            print(f"[rectify] cand{i}: rot={best['rot']} zncc={best['zncc']:.3f} -> {p2}")
    return 0


def cmd_template_build(a) -> int:
    """設計書 5.4 C1: 正規化済みの空欄サンプルからテンプレートを作る（画像は複製しない）。"""
    cfg = load_config(a.config)
    src, out = Path(a.inp).expanduser(), Path(a.out).expanduser()
    if not src.exists():
        print(f"[error] 入力が存在しません: {src}", file=sys.stderr)
        return 1
    bad = check_dirs(src, out)
    if bad:
        print(f"[error] {bad}", file=sys.stderr)
        return 1
    files = collect_inputs(src)
    if not files:
        print(f"[error] サンプル画像(jpg/jpeg/png)がありません: {src}", file=sys.stderr)
        return 1
    spec = {}
    if a.spec:
        try:
            spec = json.loads(Path(a.spec).expanduser().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[error] 測定値JSONを読めません: {a.spec}: {e}", file=sys.stderr)
            return 1
    meta = build.build_template(files, out, cfg, template_id=a.id, label=a.label, kind=a.kind, spec=spec,
                               allow_fewer=a.allow_fewer, min_samples=a.min_samples)
    print(f"[template] {out}: n_samples={meta['n_samples']} min_samples_met={meta['min_samples_met']} "
          f"align={meta['align_correlations']}")
    print(f"[template] field_rect={meta['field_rect']} anchor_type={meta['anchor_type']} "
          f"required_cues={meta['required_cues']}")
    return 0


def cmd_verify(a) -> int:
    """独立再検証: 原本・マスク済みPNG・ログだけから V1〜V6 をやり直す。"""
    cfg = load_config(a.config)
    templates, _ = load_templates(a.templates, cfg)
    log = json.loads(Path(a.log).read_text(encoding="utf-8"))
    geom = log.get("geometry")
    if not geom:
        print("[error] ログに geometry がありません（ok のログのみ検証できます）", file=sys.stderr)
        return 1
    tmpl = next((t for t in templates if t.id == geom["template_id"]), None)
    if tmpl is None:
        print(f"[error] テンプレート {geom['template_id']} がありません", file=sys.stderr)
        return 1
    orig, _ = imgio.decode(Path(a.orig).read_bytes(), int(cfg.input.max_pixels))
    masked = imgio.read_array(Path(a.masked))
    if masked is None:
        print("[error] マスク済み画像を読めません", file=sys.stderr)
        return 1
    before = {"sha256": log["input"]["sha256"], "mtime_ns": log["input"].get("mtime_ns")}
    vr = verify.verify_all(orig_arr=orig, masked_arr=masked, geom=geom, tmpl=tmpl, cfg=cfg,
                           fill_bgr=tuple(geom["fill_bgr"]), orig_path=Path(a.orig), orig_before=before)
    for k in ("V1", "V2", "V3", "V4", "V5", "V6"):
        print(f"{k}: {'pass' if vr[k]['pass'] else 'FAIL ' + str(vr[k]['code'])}")
    print("verify:", "PASS" if vr["pass"] else "FAIL")
    return 0 if vr["pass"] else 3


def build_parser():
    ap = argparse.ArgumentParser(prog="donor_mask", description="免許証裏面の臓器提供意思表示欄をマスクする（ローカル処理のみ・fail-closed）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--templates", default=str(DEFAULT_TEMPLATES))

    r = sub.add_parser("run", help="マスク処理（ファイル or ディレクトリ一括）")
    r.add_argument("--in", dest="inp", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--fill", choices=sorted(pipeline.FILL_COLORS), default="black", help="単色のみ。既定 black")
    r.add_argument("--shadow", action="store_true", help="ok 相当でも ok/ に出さず全件 review（提案をレビュー画像に描く）")
    common(r)
    rc = sub.add_parser("rectify", help="校正用: 正規化画像と5%%グリッド画像を出す（出力先はアクセス制限ディレクトリに）")
    rc.add_argument("--in", dest="inp", required=True)
    rc.add_argument("--out", required=True)
    common(rc)
    v = sub.add_parser("verify", help="独立再検証（V1〜V6）")
    v.add_argument("--orig", required=True)
    v.add_argument("--masked", required=True)
    v.add_argument("--log", required=True)
    common(v)
    tb = sub.add_parser("template-build", help="設計書 5.4 C1: 正規化済みの空欄サンプルからテンプレートを作る")
    tb.add_argument("--in", dest="inp", required=True, help="正規化済みサンプル（dir or ファイル。1011x638 のみ）")
    tb.add_argument("--out", required=True, help="templates/<id> を指す（画像は複製しない）")
    tb.add_argument("--id", required=True)
    tb.add_argument("--kind", choices=("positive", "negative"), default="positive")
    tb.add_argument("--label", default="back")
    tb.add_argument("--spec", help="測定値のJSON（field_rect / anchor_type / anchor / required_cues / "
                                  "protected_rects / ignore_rects / window_expand / expected_mask_area / note）")
    tb.add_argument("--allow-fewer", action="store_true",
                    help="min_samples 未満でも暫定テンプレートとして書く（min_samples_met=false）")
    tb.add_argument("--min-samples", type=int, default=None, help="既定は config.toml [calibrate] min_samples")
    common(tb)
    for d in DEFERRED:
        sub.add_parser(d, help="v1 では未実装（DESIGN.md 参照）")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd in DEFERRED:
        print(f"[error] {a.cmd} は v1 では未実装です（README/DESIGN の「先送り」参照）", file=sys.stderr)
        return 1
    try:
        return {"run": cmd_run, "rectify": cmd_rectify, "verify": cmd_verify,
                "template-build": cmd_template_build}[a.cmd](a)
    except ConfigError as e:
        print(f"[error] 設定/テンプレート不正: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"[error] I/O エラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
