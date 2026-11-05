"""画像の入出力。原本は読み取り専用で開く。ネットワークは使わない。"""
from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}


class InputError(Exception):
    def __init__(self, code: str, msg: str = ""):
        super().__init__(f"{code}: {msg}")
        self.code, self.msg = code, msg


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_state(path: Path) -> dict:
    """V6 用: 原本のバイト列 sha256 と mtime。"""
    data = Path(path).read_bytes()
    return {"sha256": sha256_bytes(data), "mtime_ns": os.stat(path).st_mtime_ns, "bytes": len(data)}


def sniff_format(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return None


def peek_size(data: bytes, fmt: str):
    """デコード前にヘッダから画素数を読む（解凍爆弾対策）。読めなければ None。"""
    try:
        if fmt == "png":
            return struct.unpack(">II", data[16:24])
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            mk = data[i + 1]
            if mk in (0xD8, 0x01) or 0xD0 <= mk <= 0xD7:
                i += 2
                continue
            seg = struct.unpack(">H", data[i + 2:i + 4])[0]
            if mk in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + seg
    except Exception:
        return None
    return None


def decode(data: bytes, max_pixels: int):
    """(BGR配列, 情報) を返す。形式不明=E_FORMAT_UNSUPPORTED、巨大=E_TOO_LARGE、破損=E_DECODE。

    EXIF Orientation は cv2.imdecode が反映する（IMREAD_IGNORE_ORIENTATION を付けない）。
    反映の有無は、無視フラグ付きのデコード結果と形状を比べて exif_orientation_applied に残す。
    アルファ付きPNGは白に合成する（透過部が黒になって暗い縁として誤検出されるのを避ける）。
    """
    fmt = sniff_format(data)
    if fmt is None:
        raise InputError("E_FORMAT_UNSUPPORTED", "JPEG/PNG 以外")
    size = peek_size(data, fmt)
    if size is None:
        raise InputError("E_DECODE", "ヘッダからサイズを読めない")
    if size[0] * size[1] > max_pixels:
        raise InputError("E_TOO_LARGE", f"{size[0]}x{size[1]}")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED if fmt == "png" else cv2.IMREAD_COLOR)
    if img is None:
        raise InputError("E_DECODE", "デコード失敗")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = np.clip(img[..., :3] * a + 255.0 * (1 - a) + 0.5, 0, 255).astype(np.uint8)
    if img.dtype != np.uint8:
        raise InputError("E_DECODE", f"非8bit画像: {img.dtype}")
    raw = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    applied = bool(raw is not None and raw.shape[:2] != img.shape[:2])
    info = {"format": fmt, "width": int(img.shape[1]), "height": int(img.shape[0]),
            "exif_orientation_applied": applied}
    return img, info


def read_original(path: Path, max_pixels: int):
    """(img, info, data) 日本語パス対応。"""
    data = Path(path).read_bytes()
    img, info = decode(data, max_pixels)
    return img, info, data


def encode_png(img, compression: int) -> bytes:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise OSError("PNG エンコード失敗")
    return buf.tobytes()


def encode_jpeg(img, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise OSError("JPEG エンコード失敗")
    return buf.tobytes()


def read_array(path: Path):
    """出力 PNG 読み戻し用（アルファ合成なし。3ch でなければ None）。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return img
