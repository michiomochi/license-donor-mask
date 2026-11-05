"""pytest 共通。画像は全て tests/fixtures_gen.py の合成データ（SYNTHETIC_TEST）。実在の免許証画像は使わない。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import fixtures_gen as fx  # noqa: E402

from donor_mask import imgio, pipeline  # noqa: E402
from donor_mask.config import DEFAULT_CONFIG, DEFAULT_TEMPLATES, load_config  # noqa: E402
from donor_mask.template import load_templates  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="session")
def tmpls(cfg):
    return load_templates(DEFAULT_TEMPLATES, cfg)


def run_fixture(tmp_path, fixture, cfg, tmpls, name="SYNTHETIC_TEST_in", fill="black", shadow=False):
    """合成画像をファイルに書いて 1 件処理する。(rec, out_dir, path)"""
    templates, tsha = tmpls
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    p = d / (name + fixture.ext)
    p.write_bytes(fixture.data)
    out = tmp_path / "out"
    rec, halt = pipeline.process_file(p, out, cfg, templates, tsha, fill=fill, shadow=shadow)
    assert not halt
    return rec, out, p


def ok_files(out):
    d = out / "ok"
    return sorted(f.name for f in d.glob("*") if f.is_file()) if d.exists() else []


def assert_ok_invariants(fixture, rec, fill=(0, 0, 0)):
    """設計書 6.2: ok になった件が必ず満たすこと。1つでも破れたらテスト失敗（＝誤通過）。"""
    assert rec["status"] == "ok"
    out = cv2.imdecode(np.fromfile(rec["outputs"]["masked"], np.uint8), cv2.IMREAD_COLOR)
    orig, _ = imgio.decode(fixture.data, 10**9)
    assert out.shape == orig.shape
    filled = np.all(out == np.array(fill, np.uint8), axis=2)
    # 包含率: GT_field を 1px 侵食した画素の 100% が塗り色
    fm = fx.field_mask(out.shape, fixture, erode=1)
    assert fm.sum() > 0
    cover = float((fm & filled).sum()) / float(fm.sum())
    assert cover == 1.0, f"包含率 {cover}"
    # 欄外の不変: 原本と違う画素は全て塗り色
    changed = np.any(out != orig, axis=2)
    assert not (changed & ~filled).any(), "塗り色以外の変更がある"
    # 保護領域が原本と 100% 一致
    pm = fx.protected_mask(out.shape, fixture, shrink=2)
    assert not (changed & pm).any(), "保護領域が変化した"
    # 塗りすぎ
    gt_full = fx.field_mask(out.shape, fixture, erode=0)
    assert float(changed.sum()) / float(gt_full.sum()) <= 1.6
    if fixture.gt_stroke is not None:  # はみ出し手書きの端点もマスク内
        for x, y in fixture.gt_stroke:
            assert filled[int(round(y)), int(round(x))], "手書きストロークの端点がマスク外"
    return cover


def reason_codes(rec):
    return [r["code"] for r in rec["reasons"]]
