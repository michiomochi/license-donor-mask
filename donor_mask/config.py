"""config.toml の読み込みと検証。しきい値はコードに直書きしない。"""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.toml"
DEFAULT_TEMPLATES = REPO_ROOT / "templates"


class ConfigError(Exception):
    """設定不正（CLI は終了コード 1）。"""


class Section:
    """辞書の属性アクセス用ラッパー（読み取り専用）。"""

    def __init__(self, d: dict, path: str = ""):
        self._d, self._path = d, path

    def __getattr__(self, k):
        try:
            v = self._d[k]
        except KeyError:
            raise ConfigError(f"設定キーがありません: {self._path}{k}") from None
        return Section(v, f"{self._path}{k}.") if isinstance(v, dict) else v

    def __getitem__(self, k):
        return getattr(self, k)

    def as_dict(self) -> dict:
        return self._d

    def override(self, section: str, **kv) -> "Section":
        """一部の値だけ差し替えた複製（元は変えない）。V4 の再検出で合意マップ数を緩めるのに使う。"""
        import copy
        raw = copy.deepcopy(self._d)
        raw[section].update(kv)
        return self._rebuild(raw)

    def _rebuild(self, raw):
        return Section(raw, self._path)


class Config(Section):
    def __init__(self, raw: dict, sha256: str, path: str):
        super().__init__(raw)
        self.sha256, self.source = sha256, path

    def _rebuild(self, raw):  # 差し替え後も元の設定ファイルの sha256 を保つ（ログ用。差し替えた旨は呼び出し側が記録する）
        return Config(raw, self.sha256, self.source)


# 必須キー（欠けていたら起動時に ConfigError）。値の型は数値/配列のみ許す。
_REQUIRED = {
    "input": ["max_pixels", "min_long_edge_px"],
    "canvas": ["width", "height", "card_aspect"],
    "card": ["work_long_edge", "canny_low", "canny_high", "canny_dilate", "approx_eps", "min_area_ratio",
             "max_area_ratio", "angle_min_deg", "angle_max_deg", "max_opposite_ratio", "aspect_tol",
             "edge_support_min", "edge_support_tol_px", "border_margin_ratio", "border_margin_px",
             "consensus_min_maps", "consensus_dist_ratio", "corner_refine_band", "corner_refine_min_points",
             "max_candidates"],
    "signature": ["block_size", "c", "kernel_h", "kernel_v", "out_width", "out_height", "blur_sigma"],
    "select": ["min_zncc", "min_margin", "tie_eps", "multi_card_iou"],
    "ecc": ["iterations", "eps", "blur_sigma", "gauss_filt_size", "min_correlation", "max_translation_ratio",
            "scale_min", "scale_max", "max_rotation_deg"],
    "quality": ["glare_level", "glare_max_ratio", "blur_min_laplacian_var", "contrast_min_p95_p5"],
    "field": ["frame_support_min", "frame_support_band_px", "frame_min_iou", "anchor_side_tol",
              "anchor_aspect_min", "anchor_aspect_max", "anchor_rectangularity_min", "anchor_circularity_min",
              "anchor_max_singles", "triple_side_cv_max", "triple_xshift_max", "triple_gap_ratio_max",
              "triple_pitch_tol", "iou_tf_min", "iou_fa_min", "iou_ta_min"],
    "mask": ["pad_x_min_px", "pad_x_ratio", "pad_y_min_px", "pad_y_ratio", "pad_cap_px",
             "protected_min_gap_px", "protected_clear_px", "area_ratio_min", "area_ratio_max", "grow_px"],
    "ink": ["ring_px", "ink_ratio_max", "cross_min_len_px", "expand_px", "max_expand_times",
            "print_dilate_px", "open_px", "border_ignore_px"],
    "verify": ["inside_min_dist_px", "field_iou_min", "card_corner_dist_ratio", "redetect_erode_px"],
    "output": ["png_compression", "preview_jpeg_quality"],
}


def load_config(path=None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    try:
        data = p.read_bytes()
        raw = tomllib.loads(data.decode("utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise ConfigError(f"設定を読めません: {p}: {e}") from e
    for sec, keys in _REQUIRED.items():
        if sec not in raw or not isinstance(raw[sec], dict):
            raise ConfigError(f"設定セクションがありません: [{sec}]")
        for k in keys:
            if k not in raw[sec]:
                raise ConfigError(f"設定キーがありません: [{sec}] {k}")
            v = raw[sec][k]
            if isinstance(v, bool) or not isinstance(v, (int, float, list)):
                raise ConfigError(f"設定値が数値ではありません: [{sec}] {k} = {v!r}")
    return Config(raw, hashlib.sha256(data).hexdigest(), str(p))
