#!/usr/bin/env python3
"""【数字パッチの切り出し】anchor_type=digits123 のテンプレート資産 digit_{1,2,3}.png を作る。

入力は「正規化 1011x638（S4 の ECC 後＝テンプレート座標）の空欄カード」の PNG と、印刷数字 1/2/3 の外接矩形。
各矩形を margin px 広げた範囲をグレースケールで切り出し、digit_1.png / digit_2.png / digit_3.png として書く。
パッチは「印刷様式部品（数字の字形）の位置合わせ」用の画像で、文字認識（OCR・学習済みモデル）ではない。

  python tools/cut_digit_patches.py --in <空欄カード.png> --boxes '[[x0,y0,x1,y1],[...],[...]]' \\
      --out <テンプレートdir> --margin 5

- boxes は数字の外接矩形で、両端の画素を含む（x0..x1, y0..y1 とも inclusive）。1/2/3 の順。
  JSON 文字列、またはそれを書いたファイルのパス。
- 既存の digit_*.png は上書きしない（1枚でもあれば何も書かずにエラー終了）。
- 標準出力に中心座標（連続座標。パッチの中心）と平均ピッチを JSON で出す。template.json の anchor にそのまま使う。
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

NAMES = ("digit_1.png", "digit_2.png", "digit_3.png")


def cut_patches(gray: np.ndarray, boxes, margin: int):
    """(patches, centers)。boxes は両端含む外接矩形。中心はパッチの中心（連続座標）。"""
    H, W = gray.shape[:2]
    patches, centers = [], []
    for x0, y0, x1, y1 in boxes:
        a, b, c, d = int(x0) - margin, int(y0) - margin, int(x1) + margin + 1, int(y1) + margin + 1
        if a < 0 or b < 0 or c > W or d > H or c <= a or d <= b:
            raise ValueError(f"切り出し範囲が画像の外です: box={[x0, y0, x1, y1]} margin={margin} image={W}x{H}")
        patches.append(gray[b:d, a:c].copy())
        centers.append([(a + c) / 2.0, (b + d) / 2.0])
    return patches, centers


def encode_png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise ValueError("PNG に符号化できません")
    return buf.tobytes()


def parse_boxes(text: str):
    raw = text
    if not text.lstrip().startswith("["):      # JSON 配列でなければファイルのパスとみなす
        raw = Path(text).read_text(encoding="utf-8")
    boxes = json.loads(raw)
    if not (isinstance(boxes, list) and len(boxes) == 3
            and all(isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b) for b in boxes)):
        raise ValueError("--boxes は [[x0,y0,x1,y1] x 3] の JSON であること")
    return boxes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="正規化 1011x638 の空欄カード PNG（S4 の ECC 後）")
    ap.add_argument("--boxes", required=True, help="数字 1/2/3 の外接矩形 [[x0,y0,x1,y1] x3]（JSON 文字列またはファイル）")
    ap.add_argument("--out", required=True, help="出力先ディレクトリ（テンプレート dir）")
    ap.add_argument("--margin", type=int, default=5, help="外接矩形に足す余白 px（既定 5。anchor.margin_px と揃える）")
    a = ap.parse_args(argv)
    try:
        if a.margin < 0:
            raise ValueError("--margin は 0 以上")
        gray = cv2.imdecode(np.fromfile(a.inp, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"画像を読めません: {a.inp}")
        patches, centers = cut_patches(gray, parse_boxes(a.boxes), a.margin)
        out = Path(a.out)
        exists = [n for n in NAMES if (out / n).exists()]
        if exists:
            raise ValueError(f"既存のファイルは上書きしません: {[str(out / n) for n in exists]}")
        blobs = [encode_png(p) for p in patches]
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for n, blob in zip(NAMES, blobs):
        (out / n).write_bytes(blob)
    pitch = float(np.mean(np.diff([c[1] for c in centers])))
    print(json.dumps({"patches": list(NAMES), "margin_px": a.margin, "centers": centers, "pitch": pitch,
                      "sizes_hw": [list(p.shape) for p in patches]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
