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
    "input": ["max_pixels", "min_card_long_edge_px"],
    "canvas": ["width", "height", "card_aspect"],
    "card": ["work_long_edge", "canny_low", "canny_high", "grabcut_fallback", "grabcut_margin", "grabcut_seed_a", "grabcut_seed_b", "grabcut_iters", "canny_color_low", "canny_color_high", "edge_support_tol_color_px", "canny_dilate", "approx_eps", "min_area_ratio",
             "max_area_ratio", "angle_min_deg", "angle_max_deg", "max_opposite_ratio", "aspect_tol",
             "edge_support_min", "edge_support_tol_px", "border_margin_ratio", "border_margin_px",
             "consensus_min_maps", "consensus_dist_ratio", "corner_refine_band", "corner_refine_min_points",
             "max_candidates", "frame_filled_aspect_tol", "frame_filled_corner_px", "frame_filled_zncc_min",
             "frame_filled_inset_scales"],
    "signature": ["block_size", "c", "kernel_h", "kernel_v", "out_width", "out_height", "blur_sigma"],
    "select": ["min_zncc", "min_margin", "tie_eps", "multi_card_iou", "family_iou", "mirror_margin", "mirror_m3", "m3_min_score", "m3_margin", "m3_search_px", "m3_waives_score"],
    "ecc": ["iterations", "eps", "blur_sigma", "gauss_filt_size", "min_correlation", "refine_iters", "refine_max_move_ratio", "max_translation_ratio",
            "scale_min", "scale_max", "max_rotation_deg"],
    "quality": ["glare_level", "glare_max_ratio", "blur_min_laplacian_var", "contrast_min_p95_p5"],
    "field": ["frame_support_min", "frame_support_band_px", "frame_min_iou", "f_rules_support_min", "f_rules_tol_px",
              "anchor_side_tol",
              "anchor_aspect_min", "anchor_aspect_max", "anchor_rectangularity_min", "anchor_circularity_min",
              "anchor_max_singles", "triple_side_cv_max", "triple_xshift_max", "triple_gap_ratio_max",
              "triple_pitch_tol", "iou_tf_min", "iou_fa_min", "iou_ta_min",
              "s_digit", "digit_margin_min", "digit_pitch_tol", "digit_xshift_h", "digit_gap_ratio_max", "digit_nms_px",
              "digit_max_peaks"],
    "mask": ["pad_x_min_px", "pad_x_ratio", "pad_y_min_px", "pad_y_ratio", "pad_cap_px",
             "protected_min_gap_px", "protected_clear_px", "area_ratio_min", "area_ratio_max", "grow_px"],
    "ink": ["ring_px", "ink_ratio_max", "cross_min_len_px", "expand_px", "max_expand_times",
            "print_dilate_px", "open_px", "border_ignore_px", "ink_min_len_px", "ink_edge_margin_px",
            "ink_corner_ignore_px"],
    "verify": ["inside_min_dist_px", "field_iou_min", "card_corner_dist_ratio", "card_corner_dist_ratio_color", "redetect_erode_px",
               "digit_absent_max"],
    "calibrate": ["min_samples", "align_max_correlation", "avg_threshold"],
    "output": ["png_compression", "preview_jpeg_quality"],
}

# 文字列で持つ設定キー（列挙値）。数値・bool を許すと「無効化」できてしまうものだけをここに置く。
_STRING_KEYS = {("select", "mirror_m3"): ("unimplemented", "digits23")}


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
            allowed = _STRING_KEYS.get((sec, k))
            if allowed is not None:  # 文字列で持つキー（無効化できない列挙値）は数値にできない
                if not isinstance(v, str) or v not in allowed:
                    raise ConfigError(f"設定値は {list(allowed)} のいずれかです: [{sec}] {k} = {v!r}")
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float, list)):
                raise ConfigError(f"設定値が数値ではありません: [{sec}] {k} = {v!r}")
    _validate_mirror_gate(raw)
    _validate_ink_gate(raw)
    _validate_frame_gate(raw)
    return Config(raw, hashlib.sha256(data).hexdigest(), str(p))


def _validate_frame_gate(raw: dict):
    """C_FRAME_FILLED の照合条件。範囲外の値は「常に枠いっぱい／常に違う」になる設定なので起動時に落とす。

    - frame_filled_zncc_min: (0, 1)
    - frame_filled_inset_scales: 空でない list。各要素は 0.5 < s < 1.0 の数で、厳密に降順（評価順を固定する）
    """
    c = raw["card"]
    z = c["frame_filled_zncc_min"]
    if isinstance(z, (list, bool)) or not (0.0 < float(z) < 1.0):
        raise ConfigError(f"[card] frame_filled_zncc_min は (0, 1): {z!r}")
    sc = c["frame_filled_inset_scales"]
    if not isinstance(sc, list) or not sc:
        raise ConfigError(f"[card] frame_filled_inset_scales は空でない配列: {sc!r}")
    for s in sc:
        if isinstance(s, (bool, list)) or not isinstance(s, (int, float)) or not (0.5 < float(s) < 1.0):
            raise ConfigError(f"[card] frame_filled_inset_scales の各要素は 0.5 < s < 1.0 の数: {sc!r}")
    if any(float(a) <= float(b) for a, b in zip(sc, sc[1:])):
        raise ConfigError(f"[card] frame_filled_inset_scales は厳密な降順（評価順を固定する）: {sc!r}")


def _validate_ink_gate(raw: dict):
    """V7（マスク外のインク）のしきい値も「無効にできない」。緩める設定は起動時に ConfigError。

    - ink_min_len_px: 整数 8〜48。24 が設計 §3 V7 の値。8 未満は印刷ノイズを拾い、48 超は
      「正規化キャンバスの短辺 638px に対して長すぎる」＝検出を事実上止めるため許さない
    - ink_edge_margin_px: 整数 0〜24。6 が設計値。広げるほどカード縁の成分を数えなくなる（＝緩む）
    """
    k = raw["ink"]
    lo = k["ink_min_len_px"]
    if not isinstance(lo, int) or isinstance(lo, bool) or not (8 <= lo <= 48):
        raise ConfigError(f"[ink] ink_min_len_px は 8〜48 の整数（設計値 24。小さすぎると誤検出、"
                          f"大きすぎると V7 が実質無効）: {lo!r}")
    em = k["ink_edge_margin_px"]
    if not isinstance(em, int) or isinstance(em, bool) or not (0 <= em <= 24):
        raise ConfigError(f"[ink] ink_edge_margin_px は 0〜24 の整数（設計値 6。広げるほど V7 が緩む）: {em!r}")
    cn = k["ink_corner_ignore_px"]
    if not isinstance(cn, int) or isinstance(cn, bool) or not (0 <= cn <= 120):
        raise ConfigError(f"[ink] ink_corner_ignore_px は 0〜120 の整数（設計値 48。四隅の丸い角の除外幅）: {cn!r}")


def _validate_mirror_gate(raw: dict):
    """§1.5c の鏡像ゲート M2・M3 は「無効にできない」。緩める設定は起動時に ConfigError。

    - M2（d_mirror ≥ mirror_margin）: 0 以下にできない
    - M3（印刷数字の左右反転照合）: v2.1 の一部未実装なので `unimplemented` だけを許す。
      黙って無効化する値（off / false / 0 など）は入れられない
    - 配置ファミリー family_iou も (0, 1] の外は不可
    """
    s, f = raw["select"], raw["field"]
    if not float(s["mirror_margin"]) > 0:
        raise ConfigError(f"[select] mirror_margin は正の数（M2 を無効化できない）: {s['mirror_margin']!r}")
    if not (0.0 < float(s["family_iou"]) <= 1.0):
        raise ConfigError(f"[select] family_iou は (0, 1]: {s['family_iou']!r}")
    if s["mirror_m3"] not in ("unimplemented", "digits23"):
        raise ConfigError(f"[select] mirror_m3 は \"unimplemented\" か \"digits23\" のみ（無効化・別の値は不可）"
                          f": {s['mirror_m3']!r}")
    if not (0.0 < float(f["f_rules_support_min"]) <= 1.0):
        raise ConfigError(f"[field] f_rules_support_min は (0, 1]: {f['f_rules_support_min']!r}")
    if int(f["f_rules_tol_px"]) < 0:
        raise ConfigError(f"[field] f_rules_tol_px は 0 以上: {f['f_rules_tol_px']!r}")
    # A(digits123)。範囲外の値は「数字の照合を事実上無効にする／何でも3連にする」設定になるので起動時に落とす
    if not (0.0 < float(f["s_digit"]) <= 1.0):
        raise ConfigError(f"[field] s_digit は (0, 1]: {f['s_digit']!r}")
    if not (0.0 <= float(f["digit_margin_min"]) < 1.0):
        raise ConfigError(f"[field] digit_margin_min は [0, 1): {f['digit_margin_min']!r}")
    if not (0.0 < float(f["digit_pitch_tol"]) < 1.0):
        raise ConfigError(f"[field] digit_pitch_tol は (0, 1): {f['digit_pitch_tol']!r}")
    if not float(f["digit_xshift_h"]) > 0:
        raise ConfigError(f"[field] digit_xshift_h は正の数: {f['digit_xshift_h']!r}")
    if not float(f["digit_gap_ratio_max"]) >= 1.0:
        raise ConfigError(f"[field] digit_gap_ratio_max は 1 以上: {f['digit_gap_ratio_max']!r}")
    if not isinstance(f["digit_nms_px"], int) or isinstance(f["digit_nms_px"], bool) or f["digit_nms_px"] < 1:
        raise ConfigError(f"[field] digit_nms_px は 1 以上の整数: {f['digit_nms_px']!r}")
    if not isinstance(f["digit_max_peaks"], int) or isinstance(f["digit_max_peaks"], bool) or f["digit_max_peaks"] < 1:
        raise ConfigError(f"[field] digit_max_peaks は 1 以上の整数: {f['digit_max_peaks']!r}")
    v = raw["verify"]
    if not (0.0 < float(v["digit_absent_max"]) < 1.0):
        raise ConfigError(f"[verify] digit_absent_max は (0, 1): {v['digit_absent_max']!r}")
