"""M3（印刷数字 2・3 の左右反転照合）。画像は様式テンプレート自身の印刷画像（実写真ではない）。"""
import cv2
import numpy as np
import pytest

from donor_mask import orient


@pytest.fixture
def real_tmpl(tmpls):
    return next(t for t in tmpls[0] if t.id == "real_back_v1")


def _norm(real_tmpl):
    img = 255 - cv2.imread("templates/real_back_v1/print.png", 0)  # print.png は印字=白の画像。写真と同じ「紙が明るく印字が暗い」へ
    return cv2.cvtColor(cv2.resize(img, (1011, 638)), cv2.COLOR_GRAY2BGR)


def test_m3_passes_on_upright_and_fails_on_mirrored(cfg, real_tmpl):
    n = _norm(real_tmpl)
    ok = orient.mirror_m3(n, real_tmpl, cfg)
    ng = orient.mirror_m3(np.ascontiguousarray(n[:, ::-1]), real_tmpl, cfg)
    assert ok["status"] == "checked" and ok["passed"], ok
    assert not ng["passed"], ng


def test_m3_is_not_applicable_without_digit_patches(cfg, tmpls):
    syn = next(t for t in tmpls[0] if t.id == "synthetic_back_v1")
    assert orient.mirror_m3(np.zeros((638, 1011, 3), np.uint8), syn, cfg)["passed"] is False


def test_default_config_keeps_m3_unimplemented_and_no_waive(cfg):
    assert str(cfg.select.mirror_m3) == "unimplemented" and int(cfg.select.m3_waives_score) == 0
