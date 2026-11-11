#!/usr/bin/env python3
"""【撮影バリエーションでの評価】見本から背景・傾き・指の写り込み・縮小・ブレを自動生成し、本番と同じ経路で処理して
ok 率と誤検出（塗り位置ずれ・カード以外を塗る 他）を数える。実行時に AI は使わず、パラメータの網は固定
（tests/variation_gen.py の build_grid）。特定の画像に合わせた調整はしない。

  python tools/eval_variations.py                      # 合成見本（SYNTHETIC_TEST）で全軸
  python tools/eval_variations.py --axes tilt,blur     # 軸を絞る
  python tools/eval_variations.py --json               # JSON で出す
  python tools/eval_variations.py --samples ./norm_samples --template templates/real_back_v1
      # 本番側: 正規化済み（1011x638）の見本ディレクトリ。欄・保護領域は --template の template.json から

出力するのは件数と率だけ（画像・ファイル名・パスは出さない。生成画像は一時ディレクトリに書いて評価後に消す）。
終了コード: 0 = 誤検出なし / 3 = 誤検出あり / 1 = 引数・入力の不備。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import variation_eval as ve  # noqa: E402
import variation_gen as vg  # noqa: E402

from donor_mask.config import DEFAULT_CONFIG, DEFAULT_TEMPLATES, ConfigError, load_config  # noqa: E402
from donor_mask.template import load_templates  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", help="正規化済み(1011x638)の見本ディレクトリ。省略時は合成見本")
    ap.add_argument("--template", help="--samples の欄・保護領域を読むテンプレートのディレクトリ（--samples のとき必須）")
    ap.add_argument("--axes", help="カンマ区切り: " + ",".join(vg.build_grid()))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--templates", default=str(DEFAULT_TEMPLATES))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    grid = vg.build_grid()
    axes = [x for x in (a.axes or "").split(",") if x] or None
    if axes and any(x not in grid for x in axes):
        print(f"[error] 未知の軸: {[x for x in axes if x not in grid]}（{list(grid)}）", file=sys.stderr)
        return 1
    try:
        if a.samples:
            if not a.template:
                print("[error] --samples には --template（欄・保護領域の出どころ）が必要です", file=sys.stderr)
                return 1
            samples = vg.load_samples(Path(a.samples).expanduser(), Path(a.template).expanduser())
        else:
            samples = [vg.synthetic_sample()]
        cfg = load_config(a.config)
        tmpls = load_templates(a.templates, cfg)
    except (ValueError, OSError, ConfigError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    res = ve.evaluate_grid(samples, grid, cfg, tmpls, axes=axes)
    summ = ve.summarize(res)
    summ["n_samples"] = len(samples)
    print(json.dumps(summ, ensure_ascii=False, indent=2) if a.json else f"見本 {len(samples)} 枚\n" + ve.format_text(summ))
    return 3 if summ["total"]["false_ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
