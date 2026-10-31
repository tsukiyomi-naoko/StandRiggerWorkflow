"""Minimal, dependency-free layered PSD writer (8-bit RGB, PackBits/RLE).

Written for StandRig, which reads PSDs with ag-psd: every layer is a normal
raster layer with its own alpha channel, folders use the standard
``lsct`` section markers, hidden layers set flag bit 1, and names are stored
both as a Pascal string and as a ``luni`` Unicode block.

Layers are given top-to-bottom like the Photoshop layer panel. A layer whose
name contains "/" is placed in folders ("Expressions/Mouth A"); consecutive
layers that share a folder path are grouped together.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["PsdLayer", "write_psd", "flatten_layers", "packbits_rows"]


@dataclass
class PsdLayer:
    name: str
    rgba: np.ndarray  # H x W x 4, uint8
    left: int = 0
    top: int = 0
    visible: bool = True
    opacity: float = 1.0  # 0..1
    blend: str = "norm"  # PSD blend key
    clipping: bool = False
    folder: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        a = np.asarray(self.rgba)
        if a.ndim != 3 or a.shape[2] not in (3, 4):
            raise ValueError(f"layer {self.name!r}: expected HxWx4 array, got {a.shape}")
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        if a.shape[2] == 3:
            a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=2)
        self.rgba = np.ascontiguousarray(a)
        if not self.folder and "/" in self.name:
            parts = [p.strip() for p in self.name.split("/") if p.strip()]
            self.folder, self.name = tuple(parts[:-1]), parts[-1] if parts else self.name


BLEND_KEYS = {
    "normal": "norm", "multiply": "mul ", "screen": "scrn", "overlay": "over",
    "darken": "dark", "lighten": "lite", "color-dodge": "div ", "color-burn": "idiv",
    "hard-light": "hLit", "soft-light": "sLit", "difference": "diff", "exclusion": "smud",
    "linear-dodge": "lddg", "linear-burn": "lbrn", "vivid-light": "vLit", "pin-light": "pLit",
    "linear-light": "lLit", "hard-mix": "hMix", "subtract": "fsub", "divide": "fdiv",
    "hue": "hue ", "saturation": "sat ", "color": "colr", "luminosity": "lum ",
}


# --------------------------------------------------------------------------- RLE

def _packbits_row(row: np.ndarray) -> bytes:
    n = row.shape[0]
    if n == 0:
        return b""
    # run boundaries
    change = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    out = bytearray()
    lit_start = None
    data = row.tobytes()

    def flush_literal(s: int, e: int) -> None:
        while s < e:
            c = min(128, e - s)
            out.append(c - 1)
            out.extend(data[s:s + c])
            s += c

    for s, e in zip(starts.tolist(), ends.tolist()):
        length = e - s
        if length >= 3:
            if lit_start is not None:
                flush_literal(lit_start, s)
                lit_start = None
            v = data[s]
            while length > 0:
                c = min(128, length)
                if c >= 2:
                    out.append(257 - c)
                    out.append(v)
                else:
                    out.append(0)
                    out.append(v)
                length -= c
        elif lit_start is None:
            lit_start = s
    if lit_start is not None:
        flush_literal(lit_start, n)
    return bytes(out)


def packbits_rows(channel: np.ndarray) -> tuple[List[int], bytes]:
    """PackBits-encode each row of a 2-D uint8 array. Returns (row byte counts, data)."""
    counts: List[int] = []
    chunks: List[bytes] = []
    h = channel.shape[0]
    if h == 0 or channel.shape[1] == 0:
        return [0] * h, b""
    # Fast path for constant rows (transparent areas dominate cropped layers).
    const = np.all(channel == channel[:, :1], axis=1)
    w = channel.shape[1]
    const_cache: dict = {}
    for y in range(h):
        if const[y]:
            v = int(channel[y, 0])
            enc = const_cache.get(v)
            if enc is None:
                buf = bytearray()
                left = w
                while left > 0:
                    c = min(128, left)
                    if c >= 2:
                        buf += bytes((257 - c, v))
                    else:
                        buf += bytes((0, v))
                    left -= c
                enc = const_cache[v] = bytes(buf)
        else:
            enc = _packbits_row(channel[y])
        counts.append(len(enc))
        chunks.append(enc)
    return counts, b"".join(chunks)


def _encode_channel(channel: np.ndarray) -> bytes:
    counts, data = packbits_rows(channel)
    if counts and max(counts) > 0xFFFF:  # cannot happen for <= 30000px wide PSD rows
        raise ValueError("row too long for PSD RLE")
    return struct.pack(">H", 1) + struct.pack(f">{len(counts)}H", *counts) + data


# ------------------------------------------------------------------- utilities

def _pascal(name: str, pad: int = 4) -> bytes:
    raw = name.encode("ascii", "replace")[:255]
    b = bytes([len(raw)]) + raw
    while len(b) % pad:
        b += b"\0"
    return b


def _block(key: bytes, payload: bytes, pad: int = 2) -> bytes:
    while len(payload) % pad:
        payload += b"\0"
    return b"8BIM" + key + struct.pack(">I", len(payload)) + payload


def _luni(name: str) -> bytes:
    enc = name.encode("utf-16-be")
    return _block(b"luni", struct.pack(">I", len(enc) // 2) + enc, pad=4)


def _trim(layer: PsdLayer) -> PsdLayer:
    """Crop to non-transparent bounds and zero colour under fully transparent pixels."""
    a = layer.rgba
    alpha = a[..., 3]
    ys, xs = np.nonzero(alpha)
    if ys.size == 0:
        return PsdLayer(layer.name, np.zeros((0, 0, 4), np.uint8), layer.left, layer.top,
                        layer.visible, layer.opacity, layer.blend, layer.clipping, tuple(layer.folder))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = a[y0:y1, x0:x1].copy()
    c[c[..., 3] == 0, :3] = 0
    return PsdLayer(layer.name, c, layer.left + int(x0), layer.top + int(y0),
                    layer.visible, layer.opacity, layer.blend, layer.clipping, tuple(layer.folder))


@dataclass
class _Record:
    name: str
    rgba: Optional[np.ndarray]
    left: int
    top: int
    visible: bool
    opacity: float
    blend: str
    clipping: bool
    section: int = 0  # 0 normal, 1 open folder, 3 divider


def _records_bottom_up(layers_top_down: Sequence[PsdLayer]) -> List[_Record]:
    """Turn a top-down list with folder paths into PSD bottom-up records."""
    # Build a tree preserving order.
    root: list = []
    stack: List[tuple] = [((), root)]
    for lyr in layers_top_down:
        path = tuple(lyr.folder)
        # pop until current path is a prefix of the new path
        while stack[-1][0] != path[: len(stack[-1][0])]:
            stack.pop()
        while len(stack[-1][0]) < len(path):
            depth = len(stack[-1][0])
            folder = {"folder": path[depth], "children": []}
            stack[-1][1].append(folder)
            stack.append((path[: depth + 1], folder["children"]))
        stack[-1][1].append(lyr)

    out: List[_Record] = []

    def emit(nodes: list) -> None:
        # nodes are top-down; PSD wants bottom-up
        for node in reversed(nodes):
            if isinstance(node, dict):
                out.append(_Record("</Layer group>", None, 0, 0, True, 1.0, "norm", False, 3))
                emit(node["children"])
                out.append(_Record(node["folder"], None, 0, 0, True, 1.0, "pass", False, 1))
            else:
                t = _trim(node)
                out.append(_Record(t.name, t.rgba, t.left, t.top, t.visible, t.opacity,
                                   BLEND_KEYS.get(t.blend, t.blend if len(t.blend) == 4 else "norm"),
                                   t.clipping))

    emit(list(layers_top_down if not root else root))
    return out


# ------------------------------------------------------------------ composite

def flatten_layers(layers_top_down: Sequence[PsdLayer], width: int, height: int,
                   background: Optional[tuple] = None) -> np.ndarray:
    """Normal-blend visible layers into an RGBA uint8 image (for previews / composite)."""
    canvas = np.zeros((height, width, 4), np.float32)
    if background is not None:
        canvas[..., :3] = np.asarray(background[:3], np.float32) / 255.0
        canvas[..., 3] = 1.0
    for lyr in reversed(list(layers_top_down)):
        if not lyr.visible or lyr.rgba.size == 0:
            continue
        h, w = lyr.rgba.shape[:2]
        x0, y0 = max(lyr.left, 0), max(lyr.top, 0)
        x1, y1 = min(lyr.left + w, width), min(lyr.top + h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        src = lyr.rgba[y0 - lyr.top:y1 - lyr.top, x0 - lyr.left:x1 - lyr.left].astype(np.float32) / 255.0
        sa = src[..., 3:4] * float(lyr.opacity)
        dst = canvas[y0:y1, x0:x1]
        da = dst[..., 3:4]
        oa = sa + da * (1 - sa)
        rgb = np.where(oa > 0, (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6), 0)
        dst[..., :3] = rgb
        dst[..., 3:4] = oa
    return np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------- writer

def write_psd(path: str, width: int, height: int, layers_top_down: Iterable[PsdLayer],
              composite_background: tuple = (255, 255, 255)) -> dict:
    """Write a layered PSD. Returns a small report dict."""
    layers_top_down = list(layers_top_down)
    if width <= 0 or height <= 0 or width > 30000 or height > 30000:
        raise ValueError(f"invalid PSD size {width}x{height}")
    records = _records_bottom_up(layers_top_down)

    rec_bytes = bytearray()
    chan_bytes = bytearray()
    for r in records:
        if r.rgba is None or r.rgba.size == 0:
            top = left = bottom = right = 0
            if r.rgba is not None:
                top, left, bottom, right = r.top, r.left, r.top, r.left
            empty = struct.pack(">H", 0)  # raw, zero bytes
            chans = [(-1, empty), (0, empty), (1, empty), (2, empty)]
        else:
            h, w = r.rgba.shape[:2]
            top, left, bottom, right = r.top, r.left, r.top + h, r.left + w
            chans = [(-1, _encode_channel(r.rgba[..., 3])), (0, _encode_channel(r.rgba[..., 0])),
                     (1, _encode_channel(r.rgba[..., 1])), (2, _encode_channel(r.rgba[..., 2]))]
        rec = bytearray(struct.pack(">iiiiH", top, left, bottom, right, len(chans)))
        for cid, data in chans:
            rec += struct.pack(">hI", cid, len(data))
            chan_bytes += data
        flags = 0x08
        if not r.visible:
            flags |= 0x02
        if r.section:
            flags |= 0x10
        rec += b"8BIM" + r.blend.encode("ascii")[:4].ljust(4)
        rec += struct.pack(">BBBB", int(round(max(0.0, min(1.0, r.opacity)) * 255)),
                           1 if r.clipping else 0, flags, 0)
        extra = bytearray()
        extra += struct.pack(">I", 0)  # layer mask data
        extra += struct.pack(">I", 0)  # blending ranges
        extra += _pascal(r.name)
        extra += _luni(r.name)
        if r.section == 1:
            extra += _block(b"lsct", struct.pack(">I", 1) + b"8BIM" + b"pass")
        elif r.section == 3:
            extra += _block(b"lsct", struct.pack(">I", 3))
        rec += struct.pack(">I", len(extra)) + extra
        rec_bytes += rec

    layer_info = struct.pack(">h", len(records)) + bytes(rec_bytes) + bytes(chan_bytes)
    if len(layer_info) % 2:
        layer_info += b"\0"
    layer_info_block = struct.pack(">I", len(layer_info)) + layer_info
    lmi = layer_info_block + struct.pack(">I", 0)  # empty global layer mask info

    comp = flatten_layers(layers_top_down, width, height, background=composite_background)
    comp_counts: List[int] = []
    comp_data = bytearray()
    for c in range(3):
        counts, data = packbits_rows(comp[..., c])
        comp_counts += counts
        comp_data += data

    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H", 1) + b"\0" * 6)
        f.write(struct.pack(">HIIHH", 3, height, width, 8, 3))
        f.write(struct.pack(">I", 0))  # colour mode data
        f.write(struct.pack(">I", 0))  # image resources
        f.write(struct.pack(">I", len(lmi)) + lmi)
        f.write(struct.pack(">H", 1))
        f.write(struct.pack(f">{len(comp_counts)}H", *comp_counts))
        f.write(comp_data)

    return {
        "path": path,
        "width": width,
        "height": height,
        "layers": [l.name for l in layers_top_down],
        "records": len(records),
    }
