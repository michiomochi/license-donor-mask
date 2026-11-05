"""実様式テンプレート `templates/real_back_v1` の内容を固定する。

実写真・切り出し画像・変種画像・GT の JSON はリポジトリに置かない（テンプレートの
`template.json` / `ruling.png` / `print.png` だけ）。ここで固定するのは、その3ファイルが
「どう測って、どこまで fail-closed か」を表していること。

実写真は同梱しないので、このファイルのテストは画像ファイルを一切読まない。
"""
import json
from pathlib import Path

import pytest
from conftest import ROOT, run_fixture

import fixtures_gen as fx
from donor_mask import field, mask
from donor_mask.geom import rect_area
from donor_mask.template import load_templates

DIR = ROOT / "templates" / "real_back_v1"
FIELD = [40.0, 300.0, 971.0, 597.0]
PROTECTED = [[56.0, 52.0, 955.0, 298.0]]


@pytest.fixture(scope="module")
def real(cfg):
    tmpls, _ = load_templates(ROOT / "templates", cfg)
    return next(t for t in tmpls if t.id == "real_back_v1")


def test_metadata_is_the_measured_spec(real):
    m = real.meta
    assert real.kind == "positive" and real.label == "back"
    assert m["synthetic"] is False               # 実写真から作った（生成器の出力ではない）
    assert m["size"] == [1011, 638]
    assert [float(v) for v in m["field_rect"]] == FIELD
    assert m["protected_rects"] == PROTECTED
    assert m["ignore_rects"] == [FIELD]          # 手書き・スタンプが乗りうる欄＝宣言ブロック全体
    assert m["window_expand"] == 0.1


def test_anchor_is_none_and_cues_stay_fail_closed(real):
    """実様式に印刷された印（A）は無い。T だけは不可なので required_cues は F を持つ。
    F（罫線の枠）も実在しない様式なので、このテンプレートでは ok 経路が開かない（fail-closed）。"""
    m = real.meta
    assert m["anchor_type"] == "none" and m["anchor"] is None
    assert m["required_cues"] == ["T", "F"]


def test_declared_as_provisional(real):
    """同一カード3撮影なので「様式ごとに5枚以上」を満たさない。暫定であることが JSON に出ていること。"""
    m = real.meta
    assert m["n_samples"] == 3 and m["min_samples_required"] == 5
    assert m["min_samples_met"] is False
    assert len(m["align_correlations"]) == 3 and min(m["align_correlations"]) >= 0.50
    assert "暫定" in m["note"]


def test_expected_mask_area_is_derived_from_field_rect(real, cfg):
    """expected_mask_area は field_rect と [mask] 余白から導ける値（手で書き換えていない）。"""
    pads = mask.base_pads(tuple(FIELD), cfg)
    assert real.meta["expected_mask_area"] == pytest.approx(rect_area(mask.padded(tuple(FIELD), pads)), abs=0.01)


def test_only_three_files_are_committed(real):
    """サンプル画像・切り出し・GT の JSON を入れていないこと。"""
    assert sorted(p.name for p in DIR.iterdir()) == ["print.png", "ruling.png", "template.json"]
    assert sorted(json.loads((DIR / "template.json").read_text(encoding="utf-8"))) == sorted(real.meta)


def test_no_ruling_frame_can_satisfy_f(real, cfg):
    """この様式に「欄を囲む枠罫線」は無い、という実測をテンプレート自身で固定する。

    罫線マップから取れる最大の枠はカード外周（＝正規化キャンバスの縁）で、field_rect との IoU は
    面積比（約 0.43）にしかならない。IoU(T,F) >= iou_tf_min になる枠は無いので、S6 は必ず
    F_FRAME_NOT_FOUND か F_CUE_DISAGREE で review になる（＝ok 経路が開かない）。
    """
    rect, met = field.detect_frame(real.ruling, real.rect("field_rect"), cfg)
    if rect is not None:
        iou = met["iou_T"]
        assert iou < cfg.field.iou_tf_min, f"F が T と一致してしまう（ok 経路が開く）: {rect} iou={iou}"


def test_synthetic_fixture_does_not_pick_the_real_template(cfg, tmpls, tmp_path):
    """合成フィクスチャの実行で実様式テンプレートが選ばれない（誤って ok 経路に混ざらない）。"""
    rec, _, _ = run_fixture(tmp_path, fx.make({"seed": 1}), cfg, tmpls)
    assert rec["status"] == "ok"
    assert rec["stages"]["S3"]["best"]["template"].startswith("synthetic_")
