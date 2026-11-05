"""実画像の見本（記入例・解説図）での挙動確認。リポジトリには見本画像を含めないので、環境変数が無ければ skip する。

使い方: DONOR_MASK_SAMPLE_DIR=/path/to/dir pytest tests/test_real_samples.py -s
  dir には README 記載の出典から各自ダウンロードした kumaisui_driver_license.png / mhlw_intention_01.jpg を置く。

見本は「記入例・解説図」であって実物の免許証写真ではない。v1 のテンプレートは合成レイアウトだけなので、
fail-closed の設計どおり review になる（＝無理に通さない）ことを確認する。
"""
import os
from pathlib import Path

import pytest
from conftest import ok_files

from donor_mask import pipeline

SAMPLE_DIR = os.environ.get("DONOR_MASK_SAMPLE_DIR")
NAMES = ["kumaisui_driver_license.png", "mhlw_intention_01.jpg"]

pytestmark = pytest.mark.skipif(not SAMPLE_DIR, reason="DONOR_MASK_SAMPLE_DIR 未設定（見本画像はリポジトリに含めない）")


@pytest.mark.parametrize("name", NAMES)
def test_real_sample_fails_closed(cfg, tmpls, tmp_path, name):
    p = Path(SAMPLE_DIR) / name
    if not p.exists():
        pytest.skip(f"{p} がありません")
    t, ts = tmpls
    rec, _ = pipeline.process_file(p, tmp_path / "out", cfg, t, ts)
    print(name, rec["status"], [(r["code"], r["metric"], r["value"]) for r in rec["reasons"]])
    assert rec["status"] == "review" and ok_files(tmp_path / "out") == []
    assert rec["original_unchanged"] is True


@pytest.mark.parametrize("name", NAMES)
def test_real_sample_diagnostic_run_reaches_card_detection(cfg, tmpls, tmp_path, name):
    """診断: 解像度ゲートを下げても、カード四辺が一意に取れず review（全体をカードとみなすフォールバックが無いこと）。"""
    p = Path(SAMPLE_DIR) / name
    if not p.exists():
        pytest.skip(f"{p} がありません")
    t, ts = tmpls
    diag = cfg.override("input", min_long_edge_px=200)
    rec, _ = pipeline.process_file(p, tmp_path / "out", diag, t, ts)
    print(name, rec["status"], [(r["code"], r["metric"], r["value"]) for r in rec["reasons"]])
    assert rec["status"] == "review" and ok_files(tmp_path / "out") == []
    assert "C_NO_CARD" in [r["code"] for r in rec["reasons"]]
