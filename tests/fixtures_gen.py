"""【テスト用の合成データ生成器】免許証裏面「風」の架空カードと、撮影条件の変換を決定的に作る。

重要:
  * ここで描くカードは SYNTHETIC_TEST（合成）であり、実在の免許証・個人情報を一切含まない。
    文字は全てダミー（"SYNTHETIC_TEST" 等）、番号も無い。実物の様式を再現したものでもない。
  * 欄の位置・チェックボックス3連・保護領域は「このツールの想定レイアウト」をそのまま写したもの。
    テストが示すのは「想定レイアウトなら、回転・遠近・台紙・上下逆などでも静かに誤マスクしない」ことまでで、
    実物の様式が想定どおりかは別問題（templates/ を実物から校正する必要がある）。
  * 乱数は固定シード（numpy.random.default_rng(seed)）のみ。本番コード（donor_mask/）からは使わない。
  * 生成した画像はコミットしない。テストのたびに決定的に再生成する。

座標は正規化キャンバス 1011x638 の画素座標（描画は 2 倍の 2022x1276）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

NW, NH = 1011, 638
SS = 2  # 描画の倍率（2022x1276）

# ---- 想定レイアウト（正規化座標 x0,y0,x1,y1）
FIELD_RECT = (330, 340, 950, 610)
PROTECTED = [(24, 330, 290, 600)]
IGNORE = [(330, 40, 950, 300)]
BOX_SIDE, BOX_PITCH = 28, 60
BOX_X = 352
BOX_YS = (395, 455, 515)
BOX_CENTERS = [(BOX_X + BOX_SIDE / 2, y + BOX_SIDE / 2) for y in BOX_YS]

INK = (40, 50, 40)
CARD_COLOR = (226, 238, 232)  # BGR
SKIN = (150, 175, 225)
PEN = (140, 60, 40)


def _s(v):
    return int(round(v * SS))


def _rect(img, r, t=3):
    cv2.rectangle(img, (_s(r[0]), _s(r[1])), (_s(r[2]), _s(r[3])), INK, _s(t))


def _hline(img, x0, x1, y, t=3):
    cv2.line(img, (_s(x0), _s(y)), (_s(x1), _s(y)), INK, _s(t))


def _vline(img, x, y0, y1, t=3):
    cv2.line(img, (_s(x), _s(y0)), (_s(x), _s(y1)), INK, _s(t))


def _text(img, s, x, y, scale=0.5):
    cv2.putText(img, s, (_s(x), _s(y)), cv2.FONT_HERSHEY_SIMPLEX, scale * SS, INK, max(1, SS), cv2.LINE_AA)


def draw_card(kind="back", boxes=3, extra_triple=False, color=CARD_COLOR, stroke=False, mirror=False,
              finger=False, glare=False, blank=False, outside_stroke=None) -> np.ndarray:
    """2022x1276 の BGR カード画像。kind: back（想定の裏面）/ front（表面相当）/ other（別様式）。

    blank=True は手書き・遮蔽・追加物なしの空欄（テンプレート作成用）。
    outside_stroke は「マスク（欄）の外に書かれた手書き風ストローク」の折れ線のリスト（正規化座標）。
    """
    img = np.full((NH * SS, NW * SS, 3), color, np.uint8)
    _text(img, "SYNTHETIC_TEST " + kind.upper() + " (NOT A LICENSE)", 28, 26, 0.5)
    if kind == "back":
        _rect(img, (24, 40, 290, 150))
        _text(img, "DUMMY NOTE", 34, 70, 0.45)
        for y in (185, 215, 245, 275):
            _hline(img, 24, 290, y)
        _rect(img, (330, 40, 950, 300))
        _vline(img, 560, 40, 300)
        for y in (100, 160, 220):
            _hline(img, 330, 950, y)
        _text(img, "DUMMY CHANGES", 345, 75, 0.45)
        cv2.circle(img, (_s(980), _s(90)), _s(22), INK, _s(3))
        _rect(img, PROTECTED[0])
        for y in (390, 450, 510):
            _hline(img, PROTECTED[0][0], PROTECTED[0][2], y)
        _text(img, "PROTECTED DUMMY", 34, 365, 0.45)
        _rect(img, FIELD_RECT)
        _text(img, "DUMMY FIELD TITLE", 350, 372, 0.55)
        for i in range(boxes):
            y = BOX_YS[i]
            _rect(img, (BOX_X, y, BOX_X + BOX_SIDE, y + BOX_SIDE), 3)
            _text(img, f"{i + 1} DUMMY OPTION TEXT", 400, y + 22, 0.55)
        if extra_triple:
            for y in BOX_YS:
                _rect(img, (780, y, 780 + BOX_SIDE, y + BOX_SIDE), 3)
        for x in range(352, 920, 44):  # 署名欄を模した短い破線（ruling には残らない長さ）
            _hline(img, x, x + 30, 585, 2)
        if stroke and not blank:  # 欄の外へはみ出す手書き風ストローク
            pts = np.array([[(900 + 18 * i) * SS, (505 + 6 * i) * SS] for i in range(6)], np.int32)
            cv2.polylines(img, [pts], False, PEN, _s(5), cv2.LINE_AA)
    elif kind == "front":
        _rect(img, (40, 110, 320, 520))
        for y in (130, 190, 250):
            _hline(img, 350, 970, y)
        _rect(img, (350, 300, 970, 380))
        for y in (430, 480, 530, 580):
            _hline(img, 350, 970, y)
        _rect(img, (900, 40, 970, 100))
    else:  # other
        for x in (60, 330, 600, 870):
            _vline(img, x, 60, 580)
        for y in (60, 200, 340, 480, 580):
            _hline(img, 60, 870, y)
    if not blank:
        if outside_stroke:  # 欄の外（マスクの外）に書かれた手書き風ストローク
            for pts in outside_stroke:
                arr = np.array([[int(round(x * SS)), int(round(y * SS))] for x, y in pts], np.int32)
                cv2.polylines(img, [arr], False, PEN, _s(4), cv2.LINE_AA)
        if finger:
            cv2.ellipse(img, (_s(400), _s(460)), (_s(130), _s(120)), 20, 0, 360, SKIN, -1, cv2.LINE_AA)  # 指: 左枠とチェックボックスを覆う
        if glare:
            blob = np.zeros(img.shape[:2], np.float32)
            cv2.ellipse(blob, (_s(640), _s(470)), (_s(190), _s(100)), 0, 0, 360, 1.0, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), _s(6))[..., None]
            img = np.clip(img * (1 - blob) + 255 * blob, 0, 255).astype(np.uint8)
    if mirror:
        img = img[:, ::-1].copy()
    return img


def _rounded_mask(r=28):
    m = np.zeros((NH * SS, NW * SS), np.uint8)
    rr = _s(r)
    cv2.rectangle(m, (rr, 0), (NW * SS - rr, NH * SS), 255, -1)
    cv2.rectangle(m, (0, rr), (NW * SS, NH * SS - rr), 255, -1)
    for cx, cy in ((rr, rr), (NW * SS - rr, rr), (rr, NH * SS - rr), (NW * SS - rr, NH * SS - rr)):
        cv2.circle(m, (cx, cy), rr, 255, -1)
    return m


_RM = None


def rounded_mask():
    global _RM
    if _RM is None:
        _RM = _rounded_mask()
    return _RM


# ---------------------------------------------------------------- 撮影ポーズ
PERSP_DIRS = [(0.7, 0.4), (-0.3, 0.6), (-0.5, -0.7), (0.6, -0.3)]
BASE_CW = 1500.0


def pose_quad(angle=0.0, persp=0.0, trapezoid=None):
    """スケール1.0・原点中心のカード4角（左上,右上,右下,左下）。"""
    cw, ch = BASE_CW, BASE_CW * NH / NW
    q = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]], np.float64)
    if trapezoid:  # 上辺を下辺の trapezoid 倍に（強い斜め撮影）
        q[0, 0] *= trapezoid
        q[1, 0] *= trapezoid
    if persp:
        for i, (dx, dy) in enumerate(PERSP_DIRS):
            q[i] += persp * cw * np.array([dx, dy])
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return q @ R.T


def background(kind, cw, ch, seed):
    if kind == "plain":
        return np.full((ch, cw, 3), (96, 82, 70), np.uint8)
    rng = np.random.default_rng(seed)
    low = rng.normal(0, 1, (ch // 40 + 2, cw // 40 + 2, 3)).astype(np.float32)
    low = cv2.resize(low, (cw, ch), interpolation=cv2.INTER_CUBIC) * 14
    noise = rng.normal(0, 6, (ch, cw, 3)).astype(np.float32)
    bg = np.clip(np.array((96, 82, 70), np.float32) + low + noise, 0, 255)
    if kind == "clutter":  # 明るさがカードに近い、色の違う大きなまだら（机・床・布・手）。明度の2値化では分けにくい
        n = rng.normal(0, 1, (ch // 90 + 2, cw // 90 + 2, 3)).astype(np.float32)
        n = cv2.resize(n, (cw, ch), interpolation=cv2.INTER_CUBIC)
        base = np.array((150, 170, 190), np.float32)  # BGR。カード色(クリーム)と明度が近い暖色
        bg = np.clip(base + n * np.array((35, 20, 25), np.float32) + rng.normal(0, 5, (ch, cw, 3)), 0, 255)
    if kind == "dark":  # 暗い布・机＋カード側へ向かう影
        yy, xx = np.mgrid[0:ch, 0:cw]
        grad = (1.0 - 0.5 * (xx / max(cw, 1)))[..., None]
        bg = np.clip(np.array((60, 45, 40), np.float32) * grad + low * 0.5 + noise, 0, 255)
    if kind == "desk_edge":  # 机の縁（直線）。右下側が明るい
        yy, xx = np.mgrid[0:ch, 0:cw]
        side = (xx * 0.35 + yy) > ch * 0.78
        bg[side] = np.clip(bg[side] + np.array((40, 45, 50), np.float32), 0, 255)
    return bg.astype(np.uint8)


def _warp_card(card, dst_quad, canvas_wh, canvas):
    cw, ch = canvas_wh
    tw = float(np.linalg.norm(dst_quad[1] - dst_quad[0]))
    f = min(1.0, max(tw, 200.0) / (NW * SS))
    cimg = cv2.resize(card, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else card
    rm = cv2.resize(rounded_mask(), (cimg.shape[1], cimg.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = cimg.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, np.asarray(dst_quad, np.float32))
    warped = cv2.warpPerspective(cimg, M, (cw, ch), flags=cv2.INTER_LINEAR)
    a = cv2.warpPerspective(rm, M, (cw, ch), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
    out = (warped * a + canvas * (1 - a)).astype(np.uint8)
    return out, M, f


def _map(M, f, pts_norm):
    pts = np.asarray(pts_norm, np.float64) * SS * f
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2).astype(np.float64)


@dataclass
class Fixture:
    img: np.ndarray            # 入力として使う画像（JPEG 往復後）
    data: bytes                # 入力ファイルのバイト列（.jpg / .png）
    ext: str
    quad: np.ndarray | None    # カード4角（キャンバス座標）
    gt_field: np.ndarray | None
    gt_protected: list = field(default_factory=list)
    gt_stroke: np.ndarray | None = None
    desc: str = ""


def rect_pts(r):
    return [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]


def make(case: dict | None = None, **kw) -> Fixture:
    """決定的に1枚作る。パラメータ（既定値は無加工）:

    angle, scale(0.4-1), margin(0-0.3), persp(0-0.08), trapezoid(0.5-0.7), gamma, blur, jpeg(70/95|None=PNG), seed,
    bg(plain|texture|desk_edge|paper), paper_aspect, kind(back|front|other), mirror, finger, glare, stroke, boxes,
    extra_triple, two_cards, cut_border, no_card
    """
    p = {"angle": 0.0, "scale": 1.0, "margin": 0.0, "persp": 0.0, "trapezoid": None, "gamma": 1.0, "blur": 0.0,
         "jpeg": 95, "seed": 0, "bg": "texture", "paper_aspect": None, "kind": "back", "mirror": False,
         "finger": False, "glare": False, "stroke": False, "boxes": 3, "extra_triple": False, "two_cards": False,
         "cut_border": False, "no_card": False, "card_color": CARD_COLOR, "outside_stroke": None}
    p.update(case or {})
    p.update(kw)
    card = draw_card(p["kind"], p["boxes"], p["extra_triple"], tuple(p["card_color"]), p["stroke"], p["mirror"],
                     p["finger"], p["glare"], outside_stroke=p["outside_stroke"])
    q0 = pose_quad(p["angle"], p["persp"], p["trapezoid"])
    bw, bh = np.ptp(q0[:, 0]), np.ptp(q0[:, 1])
    pad = 0.03 + p["margin"]
    cw, ch = int(round(bw * (1 + 2 * pad))), int(round(bh * (1 + 2 * pad)))
    paper = None
    if p["paper_aspect"]:
        a = p["paper_aspect"]
        cwd, chd = BASE_CW * p["scale"], BASE_CW * NH / NW * p["scale"]
        pw = max(1.25 * cwd, 1.25 * chd * a)
        ph = pw / a
        ang = np.deg2rad(p["angle"])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        paper = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]]) @ R.T
        cw = int(round(max(cw, np.ptp(paper[:, 0]) * 1.12)))
        ch = int(round(max(ch, np.ptp(paper[:, 1]) * 1.12)))
    if p["two_cards"]:
        cw = int(cw * 2.1)
    canvas = background("plain" if p["bg"] == "paper" else p["bg"], cw, ch, p["seed"])
    center = np.array([cw / 2.0, ch / 2.0])
    if p["no_card"]:
        out, quad, gt_f, gt_p, gt_s, M = canvas, None, None, [], None, None
    else:
        quad = q0 * p["scale"] + center
        if p["two_cards"]:
            quad = q0 * p["scale"] * 0.98 + np.array([cw * 0.27, ch / 2.0])
        if p["cut_border"]:
            quad = quad + np.array([-0.3 * np.ptp(quad[:, 0]) - quad[:, 0].min(), 0.0])
        if paper is not None:
            pq = (paper + center).astype(np.float32)
            cv2.fillConvexPoly(canvas, np.round(pq).astype(np.int32), (250, 250, 250), cv2.LINE_AA)
        out, M, f = _warp_card(card, quad, (cw, ch), canvas)
        if p["two_cards"]:
            q2 = q0 * p["scale"] * 0.98 + np.array([cw * 0.73, ch / 2.0])
            out, _M2, _f2 = _warp_card(card, q2, (cw, ch), out)
        mirror_x = (lambda pts: np.array([[NW - x, y] for x, y in pts])) if p["mirror"] else (lambda pts: pts)
        gt_f = _map(M, f, mirror_x(rect_pts(FIELD_RECT)))
        gt_p = [_map(M, f, mirror_x(rect_pts(r))) for r in PROTECTED]
        gt_s = _map(M, f, mirror_x([[900 + 18 * i, 505 + 6 * i] for i in range(6)])) if p["stroke"] else None
    img = out.astype(np.float32)
    if p["gamma"] != 1.0:
        img = 255.0 * (img / 255.0) ** p["gamma"]
    img = np.clip(img, 0, 255).astype(np.uint8)
    if p["blur"] > 0:
        img = cv2.GaussianBlur(img, (0, 0), p["blur"])
    if p["jpeg"]:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(p["jpeg"])])
        data, ext = buf.tobytes(), ".jpg"
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        data, ext = buf.tobytes(), ".png"
    return Fixture(img, data, ext, quad, gt_f, gt_p, gt_s, desc=str({k: v for k, v in p.items() if k != "card_color"}))


def field_mask(shape, fx: Fixture, erode=1):
    """GT_field（1px 侵食）の画素。包含率テスト用。"""
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(fx.gt_field).astype(np.int32)], 1)
    if erode:
        m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return m.astype(bool)


def protected_mask(shape, fx: Fixture, shrink=2):
    m = np.zeros(shape[:2], np.uint8)
    for p in fx.gt_protected:
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    if shrink:
        m = cv2.erode(m, np.ones((2 * shrink + 1, 2 * shrink + 1), np.uint8))
    return m.astype(bool)
