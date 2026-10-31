"""TEST-ONLY nodes standing in for GPU models so the StandRig pipeline runs on CPU.

StandRigTestFakeEdit       ~ Qwen-Image-Edit chain: shifts/tints the image per prompt
                             and stamps the prompt hash, so list mapping is visible.
StandRigTestFakeRegionMask ~ SAM3_Detect: deterministic ellipse for a few region words.
"""
import hashlib
import numpy as np
import torch
from PIL import Image, ImageDraw

REGIONS = {  # (cx, cy, rx, ry) relative to the subject box
    "head": (0.5, 0.12, 0.16, 0.12), "eyes": (0.5, 0.42, 0.28, 0.08), "mouth": (0.5, 0.62, 0.10, 0.06),
    "eyebrows": (0.5, 0.33, 0.28, 0.05), "cheeks": (0.5, 0.52, 0.32, 0.08), "face": (0.5, 0.5, 0.3, 0.35),
}


def _box(a):
    m = np.any(a < 245, axis=2)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return 0, 0, a.shape[1], a.shape[0]
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


class StandRigTestFakeEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "prompt": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "run"; CATEGORY = "StandRig/test"

    def run(self, image, prompt):
        a = (image[0].cpu().numpy() * 255).astype(np.uint8)[..., :3]
        h = int(hashlib.sha1(prompt.encode()).hexdigest()[:6], 16)
        tint = np.array([(h >> 16) & 255, (h >> 8) & 255, h & 255], np.float32)
        out = (a.astype(np.float32) * 0.85 + tint * 0.15)
        out = np.roll(out, (h % 9) - 4, axis=1)
        im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
        ImageDraw.Draw(im).text((8, im.height - 20), prompt[:60], fill=(0, 0, 0))
        return (torch.from_numpy(np.asarray(im, np.float32) / 255.0)[None],)


class StandRigTestFakeRegionMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "text": ("STRING", {"default": "head"})}}
    RETURN_TYPES = ("MASK",); FUNCTION = "run"; CATEGORY = "StandRig/test"

    def run(self, image, text):
        a = (image[0].cpu().numpy() * 255).astype(np.uint8)[..., :3]
        H, W = a.shape[:2]
        x0, y0, x1, y1 = _box(a)
        if text == "head":
            cx, cy, rx, ry = REGIONS["head"]
        else:  # face crops: relative to the whole crop
            x0, y0, x1, y1 = 0, 0, W, H
            cx, cy, rx, ry = REGIONS.get(text, REGIONS["face"])
        bw, bh = x1 - x0, y1 - y0
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).ellipse([x0 + (cx - rx) * bw, y0 + (cy - ry) * bh, x0 + (cx + rx) * bw, y0 + (cy + ry) * bh],
                                  fill=255)
        return (torch.from_numpy(np.asarray(m, np.float32) / 255.0)[None],)


NODE_CLASS_MAPPINGS = {"StandRigTestFakeEdit": StandRigTestFakeEdit,
                       "StandRigTestFakeRegionMask": StandRigTestFakeRegionMask}
