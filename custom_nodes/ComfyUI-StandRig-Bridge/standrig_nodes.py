"""ComfyUI nodes that turn generated images into StandRig inputs.

StandRig (https://github.com/sayaka-aiart/StandRig) needs, per its modeling
contract (AGENTS.md):
  * a parts-separated PSD with hidden/overlap artwork,
  * face / body / connection target sheets in a fixed panel order,
  * a ``standrig-visual-reference-v1`` manifest pinning hashes, panel order,
    parameter extremes, prompts and acceptance checks.

These nodes do the deterministic parts (prompts, crops, rotations, sheet
layout, masks, PSD writing, manifest). Image generation is done with core
ComfyUI nodes (Qwen-Image-Edit-2511) and layer decomposition with
ComfyUI-See-through or SAM3.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:  # ComfyUI runtime
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - allows unit tests without ComfyUI
    folder_paths = None

from . import prompts as P
from .psd_writer import PsdLayer, flatten_layers, write_psd

CATEGORY = "StandRig"
MAX_RES = 16384

# ----------------------------------------------------------------- tensor utils


def _t2np(img: torch.Tensor, index: int = 0) -> np.ndarray:
    """IMAGE tensor [B,H,W,C] float -> HxWxC uint8."""
    a = img[index] if img.ndim == 4 else img
    return np.clip(a.detach().cpu().float().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _np2t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)


def _mask2np(mask: Optional[torch.Tensor], h: int, w: int, union: bool = True) -> Optional[np.ndarray]:
    """MASK [B,H,W] or [H,W] -> HxW float32 0..1 resized to (h, w)."""
    if mask is None:
        return None
    m = mask.detach().cpu().float()
    if m.ndim == 2:
        m = m[None]
    if m.ndim == 4:
        m = m[..., 0] if m.shape[-1] in (1, 3, 4) else m[:, 0]
    if m.shape[0] == 0:
        return None
    m = m.max(dim=0).values if union else m[0]
    a = m.numpy()
    if a.shape != (h, w):
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    return a.astype(np.float32)


def _first(x):
    return x[0] if isinstance(x, list) else x


def _subject_bbox(rgb: np.ndarray, alpha: Optional[np.ndarray] = None, thresh: int = 245) -> Tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the subject: alpha if given, else non-white pixels."""
    if alpha is not None:
        m = alpha > 0.5
    else:
        m = np.any(rgb[..., :3] < thresh, axis=2)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return 0, 0, rgb.shape[1], rgb.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _fit_into(img: Image.Image, w: int, h: int, bg=(255, 255, 255)) -> Image.Image:
    s = min(w / img.width, h / img.height)
    nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    r = img.convert("RGB").resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGB", (w, h), bg)
    out.paste(r, ((w - nw) // 2, (h - nh) // 2))
    return out


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def _output_dir(*parts: str) -> str:
    base = folder_paths.get_output_directory() if folder_paths else os.path.abspath("output")
    safe = [re.sub(r"[^\w\-. ]+", "_", p).strip() or "untitled" for p in parts]
    d = os.path.join(base, *safe)
    os.makedirs(d, exist_ok=True)
    return d


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_png(img: Image.Image, path: str) -> Dict[str, str]:
    img.save(path, "PNG", optimize=False)
    return {"path": path, "sha256": _sha256(path)}


def _ui_image(path: str) -> Optional[dict]:
    if not folder_paths:
        return None
    out = folder_paths.get_output_directory()
    rel = os.path.relpath(path, out)
    sub, fn = os.path.split(rel)
    return {"filename": fn, "subfolder": sub, "type": "output"}


# ============================================================== 1. prepare


class StandRigPrepareCanvas:
    """Crop to the subject, scale, and lay it on a flat background.

    Output becomes the "source canvas" every later node refers to. Feed the
    foreground mask from core *Remove Background* (BiRefNet) for a clean cut.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "long_side": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 16,
                                      "tooltip": "Long side of the output canvas in pixels."}),
                "margin": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
                                     "tooltip": "Empty border around the subject, as a fraction of its size."}),
                "background": (["white", "light_gray", "black"], {"default": "white"}),
                "multiple_of": ("INT", {"default": 16, "min": 1, "max": 128}),
            },
            "optional": {
                "foreground_mask": ("MASK", {"tooltip": "1 = character. From Remove Background / SAM3."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "alpha", "rgba")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    BG = {"white": (255, 255, 255), "light_gray": (235, 235, 235), "black": (0, 0, 0)}

    def run(self, image, long_side, margin, background, multiple_of, foreground_mask=None):
        rgb = _t2np(image)[..., :3]
        h, w = rgb.shape[:2]
        alpha = _mask2np(foreground_mask, h, w)
        if alpha is None and _t2np(image).shape[2] == 4:
            alpha = _t2np(image)[..., 3].astype(np.float32) / 255.0
        x0, y0, x1, y1 = _subject_bbox(rgb, alpha)
        bw, bh = x1 - x0, y1 - y0
        m = int(round(max(bw, bh) * margin))
        if alpha is None:
            alpha = np.ones((h, w), np.float32)
        bg = self.BG[background]
        crop_rgb = Image.fromarray(rgb[y0:y1, x0:x1])
        crop_a = Image.fromarray((alpha[y0:y1, x0:x1] * 255).astype(np.uint8))
        tw, th = bw + 2 * m, bh + 2 * m
        s = long_side / max(tw, th)
        ow = max(multiple_of, int(round(tw * s / multiple_of)) * multiple_of)
        oh = max(multiple_of, int(round(th * s / multiple_of)) * multiple_of)
        s = min(ow / tw, oh / th)
        nw, nh = max(1, round(bw * s)), max(1, round(bh * s))
        ox, oy = (ow - nw) // 2, (oh - nh) // 2
        crop_rgb = crop_rgb.resize((nw, nh), Image.LANCZOS)
        crop_a = crop_a.resize((nw, nh), Image.LANCZOS)
        a = np.zeros((oh, ow), np.float32)
        a[oy:oy + nh, ox:ox + nw] = np.asarray(crop_a, np.float32) / 255.0
        fg = np.zeros((oh, ow, 3), np.float32)
        fg[oy:oy + nh, ox:ox + nw] = np.asarray(crop_rgb, np.float32)
        out = fg * a[..., None] + np.asarray(bg, np.float32) * (1 - a[..., None])
        rgba = np.concatenate([fg, a[..., None] * 255.0], axis=2)
        return (_np2t(out), torch.from_numpy(a)[None], _np2t(rgba))


# ============================================================ 2. region crop


class StandRigRegionCrop:
    """Square crop around a region (e.g. SAM3 'head' mask) for close-up edits.

    Returns crop_info so variants generated on the crop can be pasted back
    into the source canvas (StandRigVariantLayers)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 3.0, "step": 0.05,
                                      "tooltip": "Extra context around the region (fraction of its size)."}),
                "size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "fallback": (["head_from_subject_top", "center"], {"default": "head_from_subject_top"}),
                "head_fraction": ("FLOAT", {"default": 0.24, "min": 0.05, "max": 1.0, "step": 0.01,
                                            "tooltip": "Fallback only: head height as a fraction of the subject height."}),
            },
            "optional": {"region_mask": ("MASK",), "subject_alpha": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "STANDRIG_CROP", "MASK")
    RETURN_NAMES = ("crop", "crop_info", "region_mask_crop")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, padding, size, fallback, head_fraction, region_mask=None, subject_alpha=None):
        rgb = _t2np(image)[..., :3]
        H, W = rgb.shape[:2]
        rm = _mask2np(region_mask, H, W)
        if rm is not None and (rm > 0.5).any():
            ys, xs = np.nonzero(rm > 0.5)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        elif fallback == "head_from_subject_top":
            sx0, sy0, sx1, sy1 = _subject_bbox(rgb, _mask2np(subject_alpha, H, W))
            hh = (sy1 - sy0) * head_fraction
            cx = (sx0 + sx1) / 2
            # centre the head horizontally on the subject's top-most opaque pixels
            top_rows = rgb[sy0:int(sy0 + hh)]
            m = np.any(top_rows < 245, axis=2)
            if m.any():
                xs = np.nonzero(m.any(axis=0))[0]
                cx = (xs.min() + xs.max()) / 2
            x0, x1 = cx - hh / 2, cx + hh / 2
            y0, y1 = sy0, sy0 + hh
        else:
            x0, y0, x1, y1 = W * 0.25, H * 0.25, W * 0.75, H * 0.75
        side = max(x1 - x0, y1 - y0) * (1 + padding)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = int(round(min(side, max(W, H))))
        cx0 = int(round(cx - side / 2))
        cy0 = int(round(cy - side / 2))
        src = Image.fromarray(rgb)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(src.crop((cx0, cy0, cx0 + side, cy0 + side)), (0, 0))
        # PIL.crop pads outside with black; repaint outside-of-image area white
        arr = np.asarray(canvas).copy()
        ix0, iy0 = max(0, -cx0), max(0, -cy0)
        ix1, iy1 = min(side, W - cx0), min(side, H - cy0)
        mask_in = np.zeros((side, side), bool)
        mask_in[iy0:iy1, ix0:ix1] = True
        arr[~mask_in] = 255
        crop = Image.fromarray(arr).resize((size, size), Image.LANCZOS)
        info = {"x": cx0, "y": cy0, "w": side, "h": side, "src_w": W, "src_h": H, "out_w": size, "out_h": size}
        if rm is None:
            rmc = np.zeros((size, size), np.float32)
        else:
            pad = np.zeros((side, side), np.float32)
            pad[iy0:iy1, ix0:ix1] = rm[cy0 + iy0:cy0 + iy1, cx0 + ix0:cx0 + ix1]
            rmc = np.asarray(Image.fromarray((pad * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
                             np.float32) / 255.0
        return (_np2t(np.asarray(crop)), info, torch.from_numpy(rmc)[None])


# =========================================================== 3. pose prompts


PRESETS = ["face_targets", "body_targets", "connection_targets", "expressions_visemes",
           "turnaround_multiangle_lora", "turnaround_natural"]


def _build_panels(preset, subject, face_yaw, face_pitch, face_roll, body_yaw, body_pitch, body_roll,
                  roll_mode, extra, expression_list, turnaround_shot) -> List[P.Panel]:
    if preset == "face_targets":
        return P.face_targets(subject, face_yaw, face_pitch, face_roll, "ai" if roll_mode == "ai" else "rotate", extra)
    if preset == "body_targets":
        return P.body_targets(subject, body_yaw, body_pitch, body_roll, "rotate" if roll_mode == "rotate" else "ai", extra)
    if preset == "connection_targets":
        return P.connection_targets(subject, extra)
    if preset == "expressions_visemes":
        return P.expressions(subject, extra, expression_list)
    if preset == "turnaround_multiangle_lora":
        return P.turnaround(subject, extra, "multiangle_lora", turnaround_shot)
    return P.turnaround(subject, extra, "natural", turnaround_shot)


class StandRigPosePrompts:
    """Edit prompts for one reference sheet, as ComfyUI lists.

    Wire ``prompts`` into TextEncodeQwenImageEditPlus: ComfyUI runs the edit
    chain once per prompt. ``panel_ids`` / ``mask_prompts`` stay index-aligned
    with the generated images. Panels made without AI (the neutral source, exact
    roll rotations) are not in the list; StandRig Reference Sheet adds them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (PRESETS, {"default": "face_targets"}),
                "subject": ("STRING", {"default": "the character",
                                       "tooltip": "How to call the character in prompts, e.g. 'the girl with silver twin tails'."}),
                "face_yaw": ("FLOAT", {"default": 30.0, "min": 5, "max": 60, "step": 1}),
                "face_pitch": ("FLOAT", {"default": 20.0, "min": 5, "max": 45, "step": 1}),
                "face_roll": ("FLOAT", {"default": 15.0, "min": 2, "max": 45, "step": 1}),
                "body_yaw": ("FLOAT", {"default": 15.0, "min": 2, "max": 45, "step": 1}),
                "body_pitch": ("FLOAT", {"default": 8.0, "min": 2, "max": 30, "step": 1}),
                "body_roll": ("FLOAT", {"default": 8.0, "min": 2, "max": 30, "step": 1}),
                "roll_mode": (["auto", "rotate", "ai"], {"default": "auto",
                              "tooltip": "auto = exact rotation for the head, AI edit for the body (legs must stay planted)."}),
                "expression_list": ("STRING", {"default": "eyes-closed, eyes-smile, mouth-a, mouth-i, mouth-u, mouth-e, mouth-o, brows-angry, brows-sad",
                                               "multiline": True,
                                               "tooltip": "Expressions preset only. Available: " + ", ".join(e[0] for e in P.EXPRESSIONS)}),
                "turnaround_shot": (["medium shot", "wide shot", "close-up"], {"default": "wide shot"}),
                "extra": ("STRING", {"default": "", "multiline": True,
                                     "tooltip": "Appended to every prompt (style notes, outfit details)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompts", "panel_ids", "mask_prompts", "plan_json")
    OUTPUT_IS_LIST = (True, True, True, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, preset, subject, face_yaw, face_pitch, face_roll, body_yaw, body_pitch, body_roll,
            roll_mode, expression_list, turnaround_shot, extra):
        panels = _build_panels(preset, subject, face_yaw, face_pitch, face_roll, body_yaw, body_pitch, body_roll,
                               roll_mode, extra, expression_list, turnaround_shot)
        ai = [p for p in panels if p.method == "ai"]
        plan = {
            "preset": preset,
            "panels": [{"id": p.id, "method": p.method, "prompt": p.prompt, "extremes": p.extremes,
                        "direction": p.direction, "maskPrompt": p.mask_prompt} for p in panels],
            "rollDegrees": {"face": face_roll, "body": body_roll},
        }
        if not ai:  # keep downstream list nodes from receiving empty lists
            ai = [P.Panel("noop", "Return the image unchanged.")]
        return ([p.prompt for p in ai], [p.id for p in ai], [p.mask_prompt or "face" for p in ai],
                json.dumps(plan, ensure_ascii=False))


# ========================================================== 4. reference sheet


SHEET_KINDS = ["face", "body", "connections", "expressions", "turnaround"]


def _rotate_about(img: Image.Image, deg: float, pivot: Tuple[float, float], bg=(255, 255, 255)) -> Image.Image:
    # PIL rotates counter-clockwise for positive angles.
    return img.convert("RGB").rotate(deg, resample=Image.BICUBIC, center=pivot, fillcolor=bg)


def _label(img: Image.Image, text: str, sub: str = "") -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    fs = max(14, img.height // 26)
    f, fsub = _font(fs), _font(max(11, int(fs * 0.7)))
    pad = fs // 3
    tw = d.textlength(text, font=f)
    sw = d.textlength(sub, font=fsub) if sub else 0
    box_h = fs + pad * 2 + (int(fs * 0.8) if sub else 0)
    d.rectangle([0, 0, max(tw, sw) + pad * 2, box_h], fill=(20, 20, 20))
    d.text((pad, pad), text, fill=(255, 255, 255), font=f)
    if sub:
        d.text((pad, pad + fs), sub, fill=(200, 200, 200), font=fsub)
    return img


class StandRigReferenceSheet:
    """Assemble panels into a StandRig target sheet in contract order.

    Inputs arrive as lists (one entry per generated panel). The neutral panel is
    the approved source; roll panels missing from the generated list are made by
    exact rotation of the neutral image."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sheet": (SHEET_KINDS, {"default": "face"}),
                "neutral": ("IMAGE",),
                "generated": ("IMAGE",),
                "panel_ids": ("STRING", {"forceInput": True}),
                "plan_json": ("STRING", {"forceInput": True}),
                "panel_height": ("INT", {"default": 768, "min": 128, "max": 4096, "step": 16}),
                "columns": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "0 = one row."}),
                "labels": ("BOOLEAN", {"default": True}),
                "roll_pivot": (["auto", "bottom_center", "center", "hips"], {"default": "auto",
                               "tooltip": "auto: chin/neck (bottom centre) for face, hips for body."}),
                "connection_crop": ("BOOLEAN", {"default": True,
                                    "tooltip": "Connections sheet: crop each panel to its body band (relative to the subject box)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STANDRIG_SHEET")
    RETURN_NAMES = ("sheet_image", "panels", "sheet_data")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, sheet, neutral, generated, panel_ids, plan_json, panel_height, columns, labels, roll_pivot,
            connection_crop):
        sheet, panel_height, columns, labels = _first(sheet), _first(panel_height), _first(columns), _first(labels)
        roll_pivot, connection_crop = _first(roll_pivot), _first(connection_crop)
        plan = json.loads(_first(plan_json))
        neutral_img = Image.fromarray(_t2np(_first(neutral))[..., :3])

        gen: Dict[str, Image.Image] = {}
        flat_imgs: List[np.ndarray] = []
        for t in generated:
            for i in range(t.shape[0]):
                flat_imgs.append(_t2np(t, i)[..., :3])
        for pid, arr in zip(panel_ids, flat_imgs):
            gen[pid] = Image.fromarray(arr)

        plan_panels = plan.get("panels", [])
        if sheet in P.CONTRACT_ORDER:
            order = P.CONTRACT_ORDER[sheet]
        else:
            order = [p["id"] for p in plan_panels]
        by_id = {p["id"]: p for p in plan_panels}
        W, H = neutral_img.size
        nb = np.asarray(neutral_img)
        sx0, sy0, sx1, sy1 = _subject_bbox(nb)

        panels: List[Tuple[str, Image.Image, dict]] = []
        for pid in order:
            meta = dict(by_id.get(pid, {"id": pid}))
            if pid == "neutral" or meta.get("method") == "source" or (sheet == "turnaround" and pid == "front"):
                img = neutral_img
                meta["method"] = "source"
            elif pid in gen:
                img = gen[pid].resize((W, H), Image.LANCZOS) if gen[pid].size != (W, H) else gen[pid]
                meta["method"] = "ai"
            elif "roll" in pid:
                deg = float(next((v for k, v in meta.get("extremes", {}).items() if "Angle" in k), 0) or 0)
                if not deg:
                    rd = plan.get("rollDegrees", {})
                    deg = float(rd.get("body" if sheet == "body" else "face", 10)) * (-1 if pid.endswith("min") else 1)
                piv = roll_pivot
                if piv == "auto":
                    piv = "hips" if sheet == "body" else "bottom_center"
                if piv == "bottom_center":
                    pivot = ((sx0 + sx1) / 2, sy1 - (sy1 - sy0) * 0.05)
                elif piv == "hips":
                    pivot = ((sx0 + sx1) / 2, sy0 + (sy1 - sy0) * 0.55)
                else:
                    pivot = (W / 2, H / 2)
                # "-min" = crown toward viewer's left = counter-clockwise = positive PIL angle
                img = _rotate_about(neutral_img, abs(deg) if pid.endswith("min") else -abs(deg), pivot)
                meta.update({"method": "rotate", "pivot": [round(pivot[0], 1), round(pivot[1], 1)],
                             "screenRotationDegCCW": abs(deg) if pid.endswith("min") else -abs(deg)})
            else:
                img = Image.new("RGB", (W, H), (255, 255, 255))
                meta["method"] = "missing"
            if sheet == "connections" and connection_crop and meta.get("method") == "ai":
                band = (meta.get("extremes") or {}).get("cropBand")
                if band:
                    bx0 = sx0 + (sx1 - sx0) * band[0]; by0 = sy0 + (sy1 - sy0) * band[1]
                    bx1 = sx0 + (sx1 - sx0) * band[2]; by1 = sy0 + (sy1 - sy0) * band[3]
                    img = img.crop((int(bx0), int(by0), int(math.ceil(bx1)), int(math.ceil(by1))))
                    meta["crop"] = [int(bx0), int(by0), int(math.ceil(bx1)), int(math.ceil(by1))]
            panels.append((pid, img, meta))

        # uniform panel boxes
        aspect = max(p[1].width / p[1].height for p in panels)
        pw, ph = int(round(panel_height * aspect)), panel_height
        tiles = []
        for pid, img, meta in panels:
            tile = _fit_into(img, pw, ph)
            if labels:
                sub = ", ".join(f"{k}={v}" for k, v in (meta.get("extremes") or {}).items() if k != "cropBand")
                tile = _label(tile, pid, sub)
            tiles.append(tile)
        n = len(tiles)
        cols = columns if columns and columns > 0 else n
        rows = int(math.ceil(n / cols))
        gap = max(4, ph // 64)
        sheet_img = Image.new("RGB", (cols * pw + (cols + 1) * gap, rows * ph + (rows + 1) * gap), (128, 128, 128))
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet_img.paste(t, (gap + c * (pw + gap), gap + r * (ph + gap)))

        panel_batch = torch.cat([_np2t(np.asarray(_fit_into(p[1], pw, ph))) for p in panels], dim=0)
        data = {
            "kind": sheet,
            "panelOrder": [p[0] for p in panels],
            "panels": [{"id": p[0], **{k: v for k, v in p[2].items() if k != "id"}} for p in panels],
            "images": {p[0]: p[1] for p in panels},
            "sheet": sheet_img,
            "plan": plan,
        }
        return (_np2t(np.asarray(sheet_img)), panel_batch, data)


# ===================================================== 5. variant layers (PSD)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    im = Image.fromarray((mask * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.MaxFilter(px * 2 + 1)), np.float32) / 255.0


def _feather(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    im = Image.fromarray((mask * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(px)), np.float32) / 255.0


class StandRigVariantLayers:
    """Turn expression / viseme edits into hidden PSD layers.

    Each variant was generated on a close-up crop; it is masked to the region
    that changes (SAM3 mask, pixel difference against the neutral crop, or
    both), mapped back into source-canvas coordinates and returned as a core
    LAYERS document. StandRig can switch these on with opacity keys."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variants": ("IMAGE",),
                "panel_ids": ("STRING", {"forceInput": True}),
                "crop_info": ("STANDRIG_CROP",),
                "group": ("STRING", {"default": "Expressions"}),
                "mask_mode": (["sam_or_difference", "sam_mask", "difference", "sam_and_difference"],
                              {"default": "sam_or_difference"}),
                "grow_px": ("INT", {"default": 8, "min": 0, "max": 128}),
                "feather_px": ("INT", {"default": 3, "min": 0, "max": 64}),
                "diff_threshold": ("FLOAT", {"default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "region_masks": ("MASK",),
                "neutral_crop": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LAYERS", "IMAGE")
    RETURN_NAMES = ("layers", "preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, variants, panel_ids, crop_info, group, mask_mode, grow_px, feather_px, diff_threshold,
            region_masks=None, neutral_crop=None):
        info = _first(crop_info)
        group, mask_mode = _first(group), _first(mask_mode)
        grow_px, feather_px, diff_threshold = _first(grow_px), _first(feather_px), _first(diff_threshold)
        ow, oh = info["out_w"], info["out_h"]
        neutral = None
        if neutral_crop:
            neutral = np.asarray(Image.fromarray(_t2np(_first(neutral_crop))[..., :3]).resize((ow, oh), Image.LANCZOS),
                                 np.float32) / 255.0
        imgs: List[np.ndarray] = []
        for t in variants:
            for i in range(t.shape[0]):
                imgs.append(_t2np(t, i)[..., :3])
        items, previews = [], []
        for idx, (pid, arr) in enumerate(zip(panel_ids, imgs)):
            if pid in ("neutral", "noop"):
                continue
            v = np.asarray(Image.fromarray(arr).resize((ow, oh), Image.LANCZOS), np.float32) / 255.0
            sam = None
            if region_masks:
                rm = region_masks[min(idx, len(region_masks) - 1)]
                sam = _mask2np(rm, oh, ow)
                if sam is not None and not (sam > 0.5).any():
                    sam = None
            diff = None
            if neutral is not None:
                d = np.abs(v - neutral).max(axis=2)
                diff = (d > diff_threshold).astype(np.float32)
                diff = np.asarray(Image.fromarray((diff * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5)),
                                  np.float32) / 255.0
            if mask_mode == "sam_mask":
                m = sam
            elif mask_mode == "difference":
                m = diff
            elif mask_mode == "sam_and_difference":
                m = None if sam is None or diff is None else np.minimum(_dilate(sam, grow_px * 2), diff)
            else:  # sam_or_difference: prefer SAM, fall back to difference
                m = sam if sam is not None else diff
            if m is None:
                print(f"[StandRig] variant {pid}: no mask available, skipped (connect region_masks or neutral_crop)")
                continue
            m = _feather(_dilate((m > 0.5).astype(np.float32), grow_px), feather_px)
            # map crop -> source canvas
            sw, sh = info["w"], info["h"]
            rgba = np.concatenate([v * 255.0, m[..., None] * 255.0], axis=2).astype(np.uint8)
            rgba_src = np.asarray(Image.fromarray(rgba, "RGBA").resize((sw, sh), Image.LANCZOS))
            x, y = info["x"], info["y"]
            # clip to source canvas
            x0, y0 = max(0, -x), max(0, -y)
            x1, y1 = min(sw, info["src_w"] - x), min(sh, info["src_h"] - y)
            rgba_src = rgba_src[y0:y1, x0:x1]
            items.append({
                "image": torch.from_numpy(rgba_src.astype(np.float32) / 255.0).unsqueeze(0),
                "type": "raster", "name": f"{group}/{pid}", "x": int(x + x0), "y": int(y + y0),
                "z_index": 100000 + len(items), "visible": False, "opacity": 1.0,
            })
            prev = (v * 0.35 + v * m[..., None] * 0.65)
            previews.append(_np2t((prev * 255).astype(np.uint8)))
        doc = {"version": 1, "canvas": (int(info["src_w"]), int(info["src_h"])), "layers": items}
        preview = torch.cat(previews, 0) if previews else torch.zeros((1, 64, 64, 3))
        return (doc, preview)


# ================================================ 6. masks -> layers (no See-through)


class StandRigMasksToLayers:
    """Fallback part separation: cut the source by SAM3 masks into a LAYERS doc.

    Unlike See-through this does NOT paint hidden areas; use it for quick tests
    or for extra parts See-through misses (accessories, props)."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "names": ("STRING", {"forceInput": True}),
                "exclusive": ("BOOLEAN", {"default": True,
                              "tooltip": "Earlier names win overlapping pixels (list top-most parts first)."}),
                "add_remainder": ("BOOLEAN", {"default": True,
                                  "tooltip": "Put pixels no mask claimed into a bottom 'Body (rest)' layer."}),
            },
            "optional": {"subject_alpha": ("MASK",)},
        }

    RETURN_TYPES = ("LAYERS",)
    RETURN_NAMES = ("layers",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, masks, names, exclusive, add_remainder, subject_alpha=None):
        rgb = _t2np(_first(image))[..., :3]
        H, W = rgb.shape[:2]
        exclusive, add_remainder = _first(exclusive), _first(add_remainder)
        alpha = _mask2np(_first(subject_alpha), H, W) if subject_alpha else None
        if alpha is None:
            alpha = np.any(rgb < 245, axis=2).astype(np.float32)
        claimed = np.zeros((H, W), np.float32)
        items = []
        n = len(names)
        for i, name in enumerate(names):
            m = _mask2np(masks[min(i, len(masks) - 1)], H, W)
            if m is None or not (m > 0.5).any():
                print(f"[StandRig] mask for '{name}' is empty, skipped")
                continue
            m = np.minimum(m, alpha)
            if exclusive:
                m = np.clip(m - claimed, 0, 1)
            claimed = np.maximum(claimed, m)
            rgba = np.concatenate([rgb, (m[..., None] * 255)], axis=2).astype(np.uint8)
            items.append({"image": torch.from_numpy(rgba.astype(np.float32) / 255.0).unsqueeze(0), "type": "raster",
                          "name": name, "x": 0, "y": 0, "z_index": n - i})
        if add_remainder:
            rest = np.clip(alpha - claimed, 0, 1)
            if (rest > 0.5).any():
                rgba = np.concatenate([rgb, rest[..., None] * 255], axis=2).astype(np.uint8)
                items.append({"image": torch.from_numpy(rgba.astype(np.float32) / 255.0).unsqueeze(0),
                              "type": "raster", "name": "Body (rest)", "x": 0, "y": 0, "z_index": 0})
        return ({"version": 1, "canvas": (W, H), "layers": items},)


class StandRigStringList:
    """Split multiline / comma text into a ComfyUI list (e.g. SAM3 prompts)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True,
                                                  "default": "front hair\nface\neyes\nmouth\nneck\nbody"}),
                             "separator": (["newline", "comma"], {"default": "newline"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("items",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, text, separator):
        parts = text.split("\n") if separator == "newline" else text.split(",")
        items = [p.strip() for p in parts if p.strip()]
        return (items or [""],)


# ======================================================= 7. LAYERS -> PSD


SEETHROUGH_NAMES = {
    "hairf": "Front Hair", "hairb": "Back Hair", "hair": "Hair", "headwear": "Headwear", "face": "Face",
    "nose": "Nose", "mouth": "Mouth", "eyes": "Eyes", "eyel": "Eye L", "eyer": "Eye R", "browl": "Eyebrow L",
    "browr": "Eyebrow R", "eyewear": "Eyewear", "earwear": "Earwear", "neck": "Neck", "topwear": "Topwear",
    "bottomwear": "Bottomwear", "legwear": "Legwear", "footwear": "Footwear", "tail": "Tail", "wings": "Wings",
    "objects": "Objects", "handwear": "Handwear",
}
SEETHROUGH_STEMS = {"eyewhite": "Eye White", "irides": "Iris", "eyelash": "Eyelash", "eyebrow": "Eyebrow",
                    "ears": "Ear", "handwear": "Hand"}


def friendly_name(tag: str) -> str:
    if tag in SEETHROUGH_NAMES:
        return SEETHROUGH_NAMES[tag]
    m = re.fullmatch(r"(.+)-([lr])", tag)
    if m and m.group(1) in SEETHROUGH_STEMS:
        return f"{SEETHROUGH_STEMS[m.group(1)]} {m.group(2).upper()}"
    return tag


def _doc_items(doc) -> List[dict]:
    items = [i for i in (doc or {}).get("layers", []) if isinstance(i, dict) and isinstance(i.get("image"), torch.Tensor)]
    return sorted(items, key=lambda i: int(i.get("z_index", 0)))


def _item_rgba(item) -> np.ndarray:
    t = item["image"]
    a = _t2np(t)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=2)
    m = item.get("mask")
    if isinstance(m, torch.Tensor):  # core semantics: mask 1 = transparent
        mm = _mask2np(m, a.shape[0], a.shape[1])
        a[..., 3] = (a[..., 3].astype(np.float32) * (1 - mm)).astype(np.uint8)
    w, h = item.get("w"), item.get("h")
    if w and h and (w, h) != (a.shape[1], a.shape[0]):
        a = np.asarray(Image.fromarray(a, "RGBA").resize((int(w), int(h)), Image.LANCZOS))
    if item.get("flip_h"):
        a = a[:, ::-1]
    if item.get("flip_v"):
        a = a[::-1]
    return np.ascontiguousarray(a)


class StandRigLayersToPSD:
    """Write a StandRig-ready layered PSD from core LAYERS documents.

    * ``layers``: the decomposition (SeeThrough Parts To Layers, StandRig Masks
      To Layers, or Create Layered Image edits).
    * ``extra_layers``: variant layers in source-canvas coordinates (StandRig
      Variant Layers). They go on top, hidden, inside folders.
    * With ``source_image`` connected, See-through's square padded canvas is
      mapped back to the source canvas so all layers line up with the sheets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("LAYERS",),
                "project": ("STRING", {"default": "my_character"}),
                "filename": ("STRING", {"default": "character"}),
                "fit_mode": (["auto", "none", "unpad_to_source"], {"default": "auto"}),
                "rename_parts": ("BOOLEAN", {"default": True,
                                 "tooltip": "See-through tags -> readable names (L/R = character's left/right)."}),
                "min_alpha_pixels": ("INT", {"default": 16, "min": 0, "max": 100000,
                                     "tooltip": "Drop layers with fewer visible pixels (StandRig rejects empty layers)."}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "extra_layers": ("LAYERS",),
                "extra_layers_2": ("LAYERS",),
                "add_reference_layer": ("BOOLEAN", {"default": False,
                                        "tooltip": "Add the flat source as a hidden top layer 'Reference/source'."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("psd_path", "preview", "layer_report")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, layers, project, filename, fit_mode, rename_parts, min_alpha_pixels, source_image=None,
            extra_layers=None, extra_layers_2=None, add_reference_layer=False):
        canvas = (layers or {}).get("canvas")
        items = _doc_items(layers)
        if not items:
            raise ValueError("StandRig Layers To PSD: the LAYERS input has no raster layers")
        if canvas:
            cw, ch = int(canvas[0]), int(canvas[1])
        else:
            cw = max(int(i.get("x", 0)) + _item_rgba(i).shape[1] for i in items)
            ch = max(int(i.get("y", 0)) + _item_rgba(i).shape[0] for i in items)
        out_w, out_h = cw, ch
        scale, offx, offy = 1.0, 0.0, 0.0
        src = None
        if source_image is not None:
            src = _t2np(source_image)[..., :3]
            sh, sw = src.shape[:2]
            square = cw == ch
            if fit_mode == "unpad_to_source" or (fit_mode == "auto" and square and (sw, sh) != (cw, ch)):
                sz = max(sw, sh)
                scale = sz / cw
                offx, offy = (sz - sw) // 2, (sz - sh) // 2
                out_w, out_h = sw, sh

        psd_layers: List[PsdLayer] = []
        report = []

        def add(item, folder_default: Tuple[str, ...] = (), hidden_default: bool = False, rename: bool = False):
            a = _item_rgba(item)
            x, y = float(item.get("x", 0)), float(item.get("y", 0))
            if scale != 1.0 and not item.get("_source_space"):
                nw, nh = max(1, round(a.shape[1] * scale)), max(1, round(a.shape[0] * scale))
                a = np.asarray(Image.fromarray(a, "RGBA").resize((nw, nh), Image.LANCZOS))
                x, y = x * scale - offx, y * scale - offy
            if int((a[..., 3] > 8).sum()) < min_alpha_pixels:
                report.append({"name": item.get("name"), "skipped": "empty"})
                return
            raw = str(item.get("name") or f"Layer {len(psd_layers) + 1}")
            parts = [p for p in raw.split("/") if p]
            name = parts[-1] if parts else raw
            folder = tuple(parts[:-1]) or folder_default
            shown = friendly_name(name) if rename else name
            visible = bool(item.get("visible", True)) and not hidden_default
            psd_layers.append(PsdLayer(shown, a, left=int(round(x)), top=int(round(y)), visible=visible,
                                       opacity=float(item.get("opacity", 1.0)),
                                       blend=str(item.get("blend_mode", "normal")), folder=folder))
            report.append({"name": shown, "source": raw, "folder": "/".join(folder), "visible": visible,
                           "left": int(round(x)), "top": int(round(y)), "width": int(a.shape[1]),
                           "height": int(a.shape[0])})

        # Build top-down list: reference, extras (hidden), then decomposition top -> bottom
        if add_reference_layer and src is not None:
            add({"image": _np2t(src), "name": "Reference/source", "x": 0, "y": 0, "visible": False,
                 "_source_space": True})
        for extra in (extra_layers_2, extra_layers):
            if not extra:
                continue
            ex_items = _doc_items(extra)
            for it in reversed(ex_items):
                it = dict(it)
                it["_source_space"] = True
                add(it, folder_default=("Variants",))
        for it in reversed(items):
            add(it, rename=rename_parts)

        out_dir = _output_dir("standrig", project)
        stem = re.sub(r"[^\w\-.]+", "_", filename) or "character"
        psd_path = os.path.join(out_dir, f"{stem}.psd")
        info = write_psd(psd_path, out_w, out_h, psd_layers)
        flat = flatten_layers(psd_layers, out_w, out_h, background=(255, 255, 255))[..., :3]
        prev_path = os.path.join(out_dir, f"{stem}_psd_preview.png")
        Image.fromarray(flat).save(prev_path)
        rep = {"psd": psd_path, "sha256": _sha256(psd_path), "width": out_w, "height": out_h,
               "fit": {"scale": scale, "offset": [offx, offy]}, "layers": report,
               "notes": ["Layers listed top to bottom.", "L/R = the character's own left/right (Live2D convention).",
                         "Hidden layers under 'Variants' are expression/viseme swaps; drive them with opacity keys."]}
        with open(os.path.join(out_dir, f"{stem}_layers.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[StandRig] wrote {psd_path} ({len(psd_layers)} layers, {out_w}x{out_h})")
        ui = {"images": [u for u in [_ui_image(prev_path)] if u], "text": [psd_path]}
        return {"ui": ui, "result": (psd_path, _np2t(flat), json.dumps(rep, ensure_ascii=False, indent=2))}


# ============================================== 8. save reference pack + manifest


ACCEPTANCE = [
    "All required panels and images exist",
    "Preserve approved source artwork",
    "No square transparency holes",
    "No detached features or connection gaps",
    "Rotation has appropriate perspective",
    "Maximum and intermediate poses visually compared",
]
SHEET_FILES = {"face": "face-target.png", "body": "body-target.png", "connections": "connection-target.png",
               "expressions": "expression-sheet.png", "turnaround": "turnaround-sheet.png"}


class StandRigSaveReferencePack:
    """Save target sheets, per-panel PNGs and reference-manifest.json.

    Output folder: ComfyUI/output/standrig/<project>/references/. Copy it next
    to your StandRig project and pin revision/semanticHash after importing the
    PSD (StandRig Pin Manifest node or tools/pin_manifest.py)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"project": ("STRING", {"default": "my_character"})},
            "optional": {
                "face_sheet": ("STANDRIG_SHEET",),
                "body_sheet": ("STANDRIG_SHEET",),
                "connection_sheet": ("STANDRIG_SHEET",),
                "expression_sheet": ("STANDRIG_SHEET",),
                "turnaround_sheet": ("STANDRIG_SHEET",),
                "psd_path": ("STRING", {"forceInput": True}),
                "source_image": ("IMAGE",),
                "notes": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, project, face_sheet=None, body_sheet=None, connection_sheet=None, expression_sheet=None,
            turnaround_sheet=None, psd_path=None, source_image=None, notes=""):
        ref_dir = _output_dir("standrig", project, "references")
        sheets_out, ui_imgs = [], []
        source_info = None
        if source_image is not None:
            sp = os.path.join(ref_dir, "source.png")
            source_info = _save_png(Image.fromarray(_t2np(source_image)[..., :3]), sp)
        for data in (face_sheet, body_sheet, connection_sheet, expression_sheet, turnaround_sheet):
            if not data:
                continue
            kind = data["kind"]
            fn = SHEET_FILES.get(kind, f"{kind}-sheet.png")
            saved = _save_png(data["sheet"], os.path.join(ref_dir, fn))
            ui = _ui_image(saved["path"])
            if ui:
                ui_imgs.append(ui)
            pdir = _output_dir("standrig", project, "references", "panels", kind)
            panels = []
            for p in data["panels"]:
                ps = _save_png(data["images"][p["id"]], os.path.join(pdir, f"{p['id']}.png"))
                panels.append({**{k: v for k, v in p.items() if k not in ("images",)},
                               "path": os.path.relpath(ps["path"], ref_dir).replace(os.sep, "/"),
                               "sha256": ps["sha256"]})
            entry = {
                "id": "connections" if kind == "connections" else kind,
                "path": fn,
                "sha256": saved["sha256"],
                "panelOrder": data["panelOrder"],
                "parameterExtremes": {p["id"]: {k: v for k, v in (p.get("extremes") or {}).items() if k != "cropBand"}
                                      for p in data["panels"]},
                "prompt": " | ".join(f"{p['id']}: {p.get('prompt') or p.get('direction') or p.get('method')}"
                                     for p in data["panels"]),
                "panels": panels,
            }
            sheets_out.append(entry)
        psd_sha = _sha256(psd_path) if psd_path and os.path.isfile(psd_path) else "REPLACE_WITH_SOURCE_SHA256"
        manifest = {
            "format": "standrig-visual-reference-v1",
            "status": "GENERATED_PENDING_VISUAL_REVIEW",
            "source": {"revision": "PIN_AFTER_PSD_IMPORT", "semanticHash": "PIN_AFTER_PSD_IMPORT",
                       "psdSha256": psd_sha, "psdPath": psd_path or None},
            "sheets": sheets_out,
            "acceptanceChecks": ACCEPTANCE,
            "visualReview": {"accepted": False, "reportPath": None, "reviewedAt": None},
            "generator": {"tool": "ComfyUI-StandRig-Bridge", "createdAt": _dt.datetime.now().astimezone().isoformat(),
                          "sourceImage": source_info, "notes": notes or None,
                          "directionConvention": "min/max directions are described from the viewer's point of view in each panel's 'direction'."},
        }
        mp = os.path.join(ref_dir, "reference-manifest.json")
        old = None
        if os.path.isfile(mp):  # a previous step of the same project: keep what this run didn't produce
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    old = json.load(f)
            except Exception:
                old = None
        if old:
            have = {e["id"] for e in sheets_out}
            manifest["sheets"] = [e for e in old.get("sheets", []) if e.get("id") not in have] + sheets_out
            if not psd_path and old.get("source"):
                manifest["source"] = old["source"]
            if source_info is None:
                manifest["generator"]["sourceImage"] = (old.get("generator") or {}).get("sourceImage")
        ids = {e["id"] for e in manifest["sheets"]}
        missing = [k for k in ("face", "body", "connections") if k not in ids]
        if missing:
            manifest["generator"]["missingRequiredSheets"] = missing
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[StandRig] reference pack saved: {mp}")
        return {"ui": {"images": ui_imgs, "text": [mp]}, "result": (mp,)}


class StandRigPinManifest:
    """After importing the PSD into StandRig, fill revision/semanticHash in the
    manifest from GET /api/context (StandRig service must be running)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"manifest_path": ("STRING", {"default": "", "forceInput": False}),
                             "standrig_url": ("STRING", {"default": "http://127.0.0.1:5180"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, manifest_path, standrig_url):
        msg = pin_manifest(manifest_path, standrig_url)
        return {"ui": {"text": [msg]}, "result": (msg,)}


def pin_manifest(manifest_path: str, standrig_url: str) -> str:
    with urllib.request.urlopen(standrig_url.rstrip("/") + "/api/context", timeout=10) as r:
        ctx = json.loads(r.read().decode("utf-8"))
    c = ctx.get("context") or {}
    if not ctx.get("ok") or "revision" not in c:
        raise RuntimeError(f"unexpected /api/context response: {str(ctx)[:300]}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)
    man["source"]["revision"] = c["revision"]
    man["source"]["semanticHash"] = c.get("semanticHash")
    man["source"]["pinnedAt"] = _dt.datetime.now().astimezone().isoformat()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return f"pinned revision {c['revision']}"


# ============================================== 9. two-step (low memory) helpers


def _project_dir(project: str, *parts: str) -> str:
    return _output_dir("standrig", project, *parts)


class StandRigSaveLayers:
    """Save a LAYERS document (e.g. expression variants) into the project so a
    later, separate run can add it to the PSD (StandRig Load Project)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"layers": ("LAYERS",), "project": ("STRING", {"default": "my_character"}),
                             "name": ("STRING", {"default": "expressions"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, layers, project, name):
        d = _project_dir(project, "layers", name)
        for f in os.listdir(d):
            if f.endswith(".png") or f == "layers.json":
                os.remove(os.path.join(d, f))
        items = []
        for i, it in enumerate(_doc_items(layers)):
            fn = f"{i:03d}.png"
            Image.fromarray(_item_rgba(it), "RGBA").save(os.path.join(d, fn))
            items.append({"file": fn, "name": it.get("name"), "x": int(it.get("x", 0)), "y": int(it.get("y", 0)),
                          "z_index": int(it.get("z_index", i)), "visible": bool(it.get("visible", True)),
                          "opacity": float(it.get("opacity", 1.0))})
        canvas = (layers or {}).get("canvas")
        with open(os.path.join(d, "layers.json"), "w", encoding="utf-8") as f:
            json.dump({"canvas": list(canvas) if canvas else None, "layers": items}, f, indent=2, ensure_ascii=False)
        print(f"[StandRig] saved {len(items)} layers to {d}")
        return {"ui": {"text": [d]}, "result": (d,)}


class StandRigLoadProject:
    """Load what an earlier run saved for this project: the exact source canvas
    (references/source.png) and saved layers (expression variants). Used by
    the low-memory step 2 so it lines up perfectly with step 1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"project": ("STRING", {"default": "my_character"}),
                             "layers_name": ("STRING", {"default": "expressions"})}}

    RETURN_TYPES = ("IMAGE", "LAYERS", "STRING")
    RETURN_NAMES = ("source_image", "layers", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, project, layers_name):
        d = _project_dir(project)
        stamp = []
        for rel in (os.path.join("references", "source.png"), os.path.join("layers", layers_name, "layers.json")):
            fp = os.path.join(d, rel)
            stamp.append(os.path.getmtime(fp) if os.path.exists(fp) else 0)
        return str(stamp)

    def run(self, project, layers_name):
        d = _project_dir(project)
        src = os.path.join(d, "references", "source.png")
        if not os.path.isfile(src):
            raise RuntimeError(f"{src} not found. Run step 1 (low-memory sheets & expressions) with the same "
                               f"project name '{project}' first.")
        img = _np2t(np.asarray(Image.open(src).convert("RGB")))
        ld = os.path.join(d, "layers", layers_name)
        meta_p = os.path.join(ld, "layers.json")
        items = []
        canvas = (img.shape[2], img.shape[1])
        if os.path.isfile(meta_p):
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("canvas"):
                canvas = tuple(meta["canvas"])
            for it in meta.get("layers", []):
                a = np.asarray(Image.open(os.path.join(ld, it["file"])).convert("RGBA"), np.float32) / 255.0
                items.append({"image": torch.from_numpy(a).unsqueeze(0), "type": "raster", "name": it.get("name"),
                              "x": it.get("x", 0), "y": it.get("y", 0), "z_index": it.get("z_index", 0),
                              "visible": it.get("visible", True), "opacity": it.get("opacity", 1.0)})
            info = f"source {canvas[0]}x{canvas[1]}, {len(items)} saved layers"
        else:
            info = f"source {canvas[0]}x{canvas[1]}, no saved layers in {ld} (PSD will have no expression variants)"
        print(f"[StandRig] project '{project}': {info}")
        return (img, {"version": 1, "canvas": canvas, "layers": items}, info)


class StandRigFreeMemory:
    """Ask ComfyUI to unload all models and clear its cache when this run ends,
    so the next workflow starts with free RAM/VRAM (same as the Unload Models
    command). Connect any output to make it run at the end."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"after": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, after=None):
        try:
            import server  # type: ignore
            q = server.PromptServer.instance.prompt_queue
            q.set_flag("unload_models", True)
            q.set_flag("free_memory", True)
            msg = "Models will be unloaded and memory freed when this run finishes."
        except Exception as e:  # not inside ComfyUI
            msg = f"could not request memory release: {e}"
        print(f"[StandRig] {msg}")
        return {"ui": {"text": [msg]}}


NODE_CLASS_MAPPINGS = {
    "StandRigPrepareCanvas": StandRigPrepareCanvas,
    "StandRigRegionCrop": StandRigRegionCrop,
    "StandRigPosePrompts": StandRigPosePrompts,
    "StandRigReferenceSheet": StandRigReferenceSheet,
    "StandRigVariantLayers": StandRigVariantLayers,
    "StandRigMasksToLayers": StandRigMasksToLayers,
    "StandRigStringList": StandRigStringList,
    "StandRigLayersToPSD": StandRigLayersToPSD,
    "StandRigSaveReferencePack": StandRigSaveReferencePack,
    "StandRigPinManifest": StandRigPinManifest,
    "StandRigSaveLayers": StandRigSaveLayers,
    "StandRigLoadProject": StandRigLoadProject,
    "StandRigFreeMemory": StandRigFreeMemory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StandRigPrepareCanvas": "StandRig Prepare Canvas",
    "StandRigRegionCrop": "StandRig Region Crop (close-up)",
    "StandRigPosePrompts": "StandRig Pose / Expression Prompts",
    "StandRigReferenceSheet": "StandRig Reference Sheet",
    "StandRigVariantLayers": "StandRig Variant Layers (expressions)",
    "StandRigMasksToLayers": "StandRig Masks To Layers",
    "StandRigStringList": "StandRig String List",
    "StandRigLayersToPSD": "StandRig Layers To PSD",
    "StandRigSaveReferencePack": "StandRig Save Reference Pack + Manifest",
    "StandRigPinManifest": "StandRig Pin Manifest (after import)",
    "StandRigSaveLayers": "StandRig Save Layers (for a later run)",
    "StandRigLoadProject": "StandRig Load Project (step 2)",
    "StandRigFreeMemory": "StandRig Free Memory After Run",
}
