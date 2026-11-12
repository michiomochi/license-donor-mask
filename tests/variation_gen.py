"""【評価用の撮影バリエーション生成】見本（正規化済みの空欄カード）から、背景・傾き・指の写り込み・縮小・ブレを
決定的に自動生成し、正解（カード4角・欄・保護領域）を同じ変換で持ち回る。

方針:
  * 乱数は固定シード（numpy.random.default_rng）のみ。本番コード（donor_mask/）は一切変えない・import もさせない。
  * 特定の画像に合わせた設定はしない。パラメータの網（GRID）は固定で、見本や結果を見て調整しない。
  * 出力は Fixture（fixtures_gen.Fixture）。画像はメモリ上のバイト列だけで、ここではファイルに書かない。
  * 見本は 1011x638 の正規化カード（template-build の入力と同じ形式。任意の解像度は幅で自動スケール）。
    正解の座標（欄・保護領域）は常に正規化座標 1011x638 で与える。合成見本は fixtures_gen.draw_card（SYNTHETIC_TEST）。

生成の順序（現実の撮影の順に近い）: 見本 → 射影（傾き・遠近・縮尺）で背景に置く → 指 → 縮小(INTER_AREA) → ブレ → JPEG。
  縮小の後にブレを掛けるのは、ブレをセンサ画素単位で与えるため（縮小で効果が薄れない）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import fixtures_gen as fx

NW, NH = fx.NW, fx.NH

# ---------------------------------------------------------------- 見本


@dataclass
class Sample:
    img: np.ndarray                     # BGR。幅 W、高さ W*NH/NW（正規化カードの縦横比）
    field_rect: tuple                   # 正規化座標 (x0,y0,x1,y1)。GT の欄
    protected_rects: list = field(default_factory=list)
    label: str = "synthetic"


def synthetic_sample() -> Sample:
    """想定レイアウトの合成見本（SYNTHETIC_TEST）。欄・保護領域は fixtures_gen の値。"""
    return Sample(fx.draw_card("back", 3, blank=True), tuple(fx.FIELD_RECT), [tuple(r) for r in fx.PROTECTED])


def load_samples(directory, template_dir) -> list[Sample]:
    """正規化済みの見本（jpg/png）のディレクトリ → Sample。欄・保護領域はテンプレートの template.json から。

    画像の内容は出力しない・複製しない（読むだけ）。1011x638 以外は読まずに ValueError。"""
    import json
    from pathlib import Path

    meta = json.loads((Path(template_dir) / "template.json").read_text(encoding="utf-8"))
    fr = tuple(float(v) for v in meta["field_rect"])
    pr = [tuple(float(v) for v in r) for r in meta.get("protected_rects", [])]
    out = []
    for p in sorted(Path(directory).iterdir(), key=lambda f: f.name):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg") or not p.is_file():
            continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[1] != NW or img.shape[0] != NH:
            raise ValueError(f"見本は正規化済み {NW}x{NH} のみ（1枚が不正: {len(out)} 枚目の次）")
        out.append(Sample(img, fr, pr, label=f"sample{len(out)}"))
    if not out:
        raise ValueError("見本画像(jpg/png)がありません")
    return out


# ---------------------------------------------------------------- 背景

BACKGROUNDS = ("plain", "texture", "desk_edge", "clutter", "dark", "wood", "gradient", "bright")


def make_background(kind: str, cw: int, ch: int, seed: int) -> np.ndarray:
    """fixtures_gen の背景 + wood（木目の縞）/ gradient（明暗の勾配）/ bright（カードより明るい白っぽい机）。"""
    if kind in ("plain", "texture", "desk_edge", "clutter", "dark"):
        return fx.background(kind, cw, ch, seed)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    noise = rng.normal(0, 4, (ch, cw, 3)).astype(np.float32)
    if kind == "wood":  # 暖色の縞（周期 ~35px）に低周波のうねり
        wob = cv2.resize(rng.normal(0, 1, (ch // 60 + 2, cw // 60 + 2)).astype(np.float32), (cw, ch), interpolation=cv2.INTER_CUBIC)
        s = 0.5 + 0.5 * np.sin((yy + 25 * wob) * 2 * np.pi / 35.0)
        bg = np.array((70, 105, 150), np.float32) * (0.75 + 0.35 * s[..., None]) + noise
    elif kind == "gradient":  # 左上が明るく右下が暗い（照明の偏り）。カードの色に近い帯を通る
        t = ((xx / max(cw, 1)) * 0.6 + (yy / max(ch, 1)) * 0.4)[..., None]
        bg = np.array((215, 215, 205), np.float32) * (1 - t) + np.array((60, 60, 60), np.float32) * t + noise
    elif kind == "bright":  # 白い机・白い紙の上（カードより明るい）
        bg = np.array((248, 248, 246), np.float32) + noise * 0.5
    else:
        raise ValueError(f"未知の背景: {kind}")
    return np.clip(bg, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 射影で置く

def _card_image(sample: Sample, target_w: int) -> np.ndarray:
    """見本を、置くカードの幅（画素）に近い大きさにしておく（拡大しない。縮小は INTER_AREA）。"""
    h, w = sample.img.shape[:2]
    f = min(1.0, max(float(target_w), 200.0) / w)
    return cv2.resize(sample.img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else sample.img


def place(sample: Sample, *, bg="texture", angle=0.0, persp=0.0, scale=1.0, margin=0.12, seed=0):
    """(canvas BGR uint8, quad(4,2), M: 見本の正規化座標→キャンバス, gt_field(4,2), gt_protected[list])。"""
    q0 = fx.pose_quad(angle, persp, None)
    bw, bh = np.ptp(q0[:, 0]), np.ptp(q0[:, 1])
    pad = 0.03 + margin
    cw, ch = int(round(bw * (1 + 2 * pad))), int(round(bh * (1 + 2 * pad)))
    canvas = make_background(bg, cw, ch, seed)
    quad = q0 * scale + np.array([cw / 2.0, ch / 2.0])
    tw = float(np.linalg.norm(quad[1] - quad[0]))
    card = _card_image(sample, int(round(tw)))
    h, w = card.shape[:2]
    # 丸角のマスク（fixtures_gen と同じ形）
    rm = cv2.resize(fx.rounded_mask(), (w, h), interpolation=cv2.INTER_AREA)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    H = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(card, H, (cw, ch), flags=cv2.INTER_LINEAR)
    a = cv2.warpPerspective(rm, H, (cw, ch), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
    canvas = (warped * a + canvas * (1 - a)).astype(np.uint8)
    # 正規化座標 → 見本画像の画素 → キャンバス
    S = np.diag([w / NW, h / NH, 1.0])
    M = H @ S

    def to_canvas(pts):
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, M).reshape(-1, 2)

    return canvas, quad, M, to_canvas(fx.rect_pts(sample.field_rect)), [to_canvas(fx.rect_pts(r)) for r in sample.protected_rects]


# ---------------------------------------------------------------- 指の写り込み

SKINS = ((150, 175, 225), (120, 150, 200), (170, 190, 235))  # BGR
FINGER_AT = ("bottom", "left", "right", "top", "field")


def add_finger(canvas, quad, gt_field, *, at="bottom", overlap=0.15, width=0.18, skin=0, seed=0):
    """肌色の楕円（指）をカードの縁から差し込む。at=field は欄の中心に置く（欄を隠す＝ok にしてはいけない側の条件）。

    overlap: カード短辺に対する、指がカードへ入り込む長さの比。width: カード短辺に対する指の太さの比。
    縁は少しぼかし、わずかなムラを足す。決定的（seed）。"""
    out = canvas.copy()
    ch, cw = out.shape[:2]
    q = np.asarray(quad, np.float64)
    short = float(min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])))
    sides = {"top": (0, 1), "right": (1, 2), "bottom": (2, 3), "left": (3, 0)}
    if at == "field":
        c = np.mean(np.asarray(gt_field, np.float64), axis=0)
        direction = np.array([1.0, 0.0])
        L = 0.45 * short
    else:
        i, j = sides[at]
        mid = (q[i] + q[j]) / 2.0
        inward = np.mean(q, axis=0) - mid
        inward /= np.linalg.norm(inward)
        L = 0.9 * short
        c = mid - inward * (L - overlap * short)  # 楕円の中心はカードの外側（指は外から来る）
        direction = inward
    ang = float(np.degrees(np.arctan2(direction[1], direction[0])))
    layer = np.zeros((ch, cw), np.float32)
    cv2.ellipse(layer, (int(round(c[0])), int(round(c[1]))), (int(round(L)), int(round(width * short / 2))), ang, 0, 360, 1.0, -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, 0.004 * short))[..., None]
    rng = np.random.default_rng(seed)
    shade = 1.0 + 0.06 * cv2.resize(rng.normal(0, 1, (ch // 50 + 2, cw // 50 + 2)).astype(np.float32), (cw, ch), interpolation=cv2.INTER_CUBIC)[..., None]
    colour = np.array(SKINS[skin % len(SKINS)], np.float32) * shade
    return np.clip(out * (1 - layer) + colour * layer, 0, 255).astype(np.uint8), layer[..., 0] > 0.5


# ---------------------------------------------------------------- 縮小・ブレ・JPEG

def shrink(img, pts_list, factor):
    """全体を factor 倍に縮小（INTER_AREA）。GT 座標も同じ倍率で返す。"""
    if factor >= 1.0:
        return img, pts_list
    out = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    # cv2.resize の画素中心の規約（x_dst = (x_src+0.5)*f - 0.5）に合わせる
    return out, [(np.asarray(p, np.float64) + 0.5) * factor - 0.5 for p in pts_list]


def blur(img, *, kind="gaussian", strength=1.0, angle=0.0):
    """kind=gaussian（σ=strength px）/ motion（長さ strength px の直線ブレ、向き angle 度）。"""
    if strength <= 0:
        return img
    if kind == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), float(strength))
    if kind == "motion":
        n = max(3, int(round(strength)) | 1)
        k = np.zeros((n, n), np.float32)
        cv2.line(k, (0, n // 2), (n - 1, n // 2), 1.0, 1)
        R = cv2.getRotationMatrix2D(((n - 1) / 2.0, (n - 1) / 2.0), float(angle), 1.0)
        k = cv2.warpAffine(k, R, (n, n), flags=cv2.INTER_LINEAR)
        return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6), borderType=cv2.BORDER_REPLICATE)
    raise ValueError(f"未知のブレ: {kind}")


# ---------------------------------------------------------------- 1枚

DEFAULTS = {"bg": "texture", "angle": 0.0, "persp": 0.0, "scale": 1.0, "margin": 0.12, "finger": None, "finger_overlap": 0.15,
            "finger_width": 0.18, "skin": 0, "shrink": 1.0, "blur": 0.0, "blur_kind": "gaussian", "blur_angle": 0.0,
            "jpeg": 90, "seed": 0}


def make_variation(sample: Sample, case: dict | None = None, **kw) -> fx.Fixture:
    """見本 1 枚から撮影バリエーション 1 枚（決定的）。case のキーは DEFAULTS を参照。未知のキーは KeyError。"""
    p = dict(DEFAULTS)
    for k in list((case or {}).keys()) + list(kw.keys()):
        if k not in DEFAULTS:
            raise KeyError(f"未知のパラメータ: {k}")
    p.update(case or {})
    p.update(kw)
    canvas, quad, _M, gt_f, gt_p = place(sample, bg=p["bg"], angle=p["angle"], persp=p["persp"], scale=p["scale"],
                                         margin=p["margin"], seed=p["seed"])
    if p["finger"]:
        canvas, _ = add_finger(canvas, quad, gt_f, at=p["finger"], overlap=p["finger_overlap"], width=p["finger_width"],
                               skin=p["skin"], seed=p["seed"])
    pts = [quad, gt_f] + list(gt_p)
    img, pts = shrink(canvas, pts, p["shrink"])
    img = blur(img, kind=p["blur_kind"], strength=p["blur"], angle=p["blur_angle"])
    if p["jpeg"]:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(p["jpeg"])])
        data, ext = buf.tobytes(), ".jpg"
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        data, ext = buf.tobytes(), ".png"
    return fx.Fixture(img, data, ext, pts[0], pts[1], pts[2:], None, desc=str(p))


# ---------------------------------------------------------------- パラメータの網（固定。結果を見て調整しない）

def build_grid() -> dict[str, list[dict]]:
    """軸ごとのケース。一因子ずつ（他は既定）＋ 複合（固定シードの乱択）。"""
    g: dict[str, list[dict]] = {}
    g["background"] = [{"bg": b, "angle": a} for b in BACKGROUNDS for a in (0, 8)]
    g["tilt"] = [{"angle": a} for a in (-30, -20, -10, -5, 5, 10, 20, 30)] + \
                [{"angle": a, "persp": p} for p in (0.04, 0.08) for a in (0, 10)]
    g["finger"] = [{"finger": at, "skin": s} for at in ("bottom", "left", "right", "top") for s in (0, 1)] + \
                  [{"finger": "field"}, {"finger": "field", "skin": 2, "angle": 8}]
    g["shrink"] = [{"shrink": s} for s in (0.8, 0.65, 0.5, 0.4, 0.3, 0.2)]
    g["blur"] = [{"blur": s} for s in (0.8, 1.5, 2.5, 4.0)] + \
                [{"blur": n, "blur_kind": "motion", "blur_angle": a} for n in (5, 11, 21) for a in (0, 45)]
    rng = np.random.default_rng(20261112)
    combos = []
    for i in range(16):
        c = {"bg": str(rng.choice(BACKGROUNDS)), "angle": float(np.round(rng.uniform(-20, 20), 1)),
             "persp": float(np.round(rng.choice([0.0, 0.03, 0.06]), 2)), "scale": float(np.round(rng.uniform(0.7, 1.0), 2)),
             "shrink": float(np.round(rng.choice([1.0, 0.8, 0.6]), 2)), "blur": float(np.round(rng.choice([0.0, 0.8, 1.5]), 1)),
             "jpeg": int(rng.choice([70, 90])), "seed": int(i)}
        if rng.random() < 0.4:
            c["finger"] = str(rng.choice(["bottom", "left", "right", "top"]))
        combos.append(c)
    g["combo"] = combos
    return g
