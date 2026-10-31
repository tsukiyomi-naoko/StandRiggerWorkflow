"""TEST-ONLY stand-in for ComfyUI-See-through (jtydhr88/ComfyUI-See-through).

Same node names, inputs and outputs as the real pack (nodes.py @ 98d754b) so
workflows can be validated and the StandRig pipeline exercised on CPU. It does
not decompose anything: PostProcess cuts the image into horizontal bands named
like See-through tags, on the same center-padded square canvas the real
pack uses, which is what StandRig Layers To PSD has to undo.
"""
import numpy as np
import torch
from PIL import Image

TAGS = [("hairb", 0.0, 0.30, 0.9), ("hairf", 0.0, 0.12, 0.1), ("face", 0.08, 0.26, 0.5),
        ("eyewhite-l", 0.14, 0.18, 0.3), ("eyewhite-r", 0.14, 0.18, 0.3), ("mouth", 0.20, 0.24, 0.2),
        ("neck", 0.24, 0.32, 0.6), ("topwear", 0.30, 0.60, 0.55), ("bottomwear", 0.58, 0.78, 0.55),
        ("legwear", 0.76, 1.0, 0.6)]


class _M:  # model placeholder
    pass


class SeeThrough_LoadLayerDiffModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": (["layerdifforg/seethroughv0.0.2_layerdiff3d", "24yearsold/seethroughv0.0.2_layerdiff3d_nf4"],
                                       {"default": "layerdifforg/seethroughv0.0.2_layerdiff3d"})},
                "optional": {"vae_ckpt": ("STRING", {"default": ""}), "unet_ckpt": ("STRING", {"default": ""}),
                             "quant_mode": (["none", "nf4"], {"default": "none"}),
                             "cache_tag_embeds": ("BOOLEAN", {"default": True}),
                             "group_offload": ("BOOLEAN", {"default": False}),
                             "auto_download": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("SEETHROUGH_LAYERDIFF_MODEL",); RETURN_NAMES = ("layerdiff_model",)
    FUNCTION = "load_model"; CATEGORY = "SeeThrough"
    def load_model(self, model, **kw):
        return (_M(),)


class SeeThrough_LoadDepthModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": (["layerdifforg/seethroughv0.0.1_marigold", "24yearsold/seethroughv0.0.1_marigold_nf4"],
                                       {"default": "layerdifforg/seethroughv0.0.1_marigold"})},
                "optional": {"quant_mode": (["none", "nf4"], {"default": "none"}),
                             "cache_tag_embeds": ("BOOLEAN", {"default": True}),
                             "group_offload": ("BOOLEAN", {"default": False}),
                             "auto_download": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("SEETHROUGH_DEPTH_MODEL",); RETURN_NAMES = ("depth_model",)
    FUNCTION = "load_model"; CATEGORY = "SeeThrough"
    def load_model(self, model, **kw):
        return (_M(),)


class SeeThrough_GenerateLayers:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "layerdiff_model": ("SEETHROUGH_LAYERDIFF_MODEL",),
                             "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                             "resolution": ("INT", {"default": 1280, "min": 512, "max": 2048, "step": 64}),
                             "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100})}}
    RETURN_TYPES = ("SEETHROUGH_LAYERS", "IMAGE"); RETURN_NAMES = ("layers", "preview")
    FUNCTION = "generate"; CATEGORY = "SeeThrough"
    def generate(self, image, layerdiff_model, seed, resolution, num_inference_steps):
        a = (image[0].cpu().numpy() * 255).astype(np.uint8)[..., :3]
        h, w = a.shape[:2]; sz = max(h, w)
        pad = np.full((sz, sz, 3), 255, np.uint8)
        px, py = (sz - w) // 2, (sz - h) // 2
        pad[py:py + h, px:px + w] = a
        full = np.asarray(Image.fromarray(pad).resize((resolution, resolution), Image.LANCZOS))
        return ({"fullpage": full, "res": resolution}, image)


class SeeThrough_GenerateDepth:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"layers": ("SEETHROUGH_LAYERS",), "depth_model": ("SEETHROUGH_DEPTH_MODEL",),
                             "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1})},
                "optional": {"resolution_depth": ("INT", {"default": -1, "min": -1, "max": 2048, "step": 64})}}
    RETURN_TYPES = ("SEETHROUGH_LAYERS_DEPTH", "IMAGE"); RETURN_NAMES = ("layers_depth", "preview")
    FUNCTION = "generate"; CATEGORY = "SeeThrough"
    def generate(self, layers, depth_model, seed, resolution_depth=-1):
        return (layers, torch.zeros((1, 64, 64, 3)))


class SeeThrough_PostProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"layers_depth": ("SEETHROUGH_LAYERS_DEPTH",),
                             "tblr_split": ("BOOLEAN", {"default": True}), "use_lama": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("SEETHROUGH_PARTS", "IMAGE"); RETURN_NAMES = ("parts", "preview")
    FUNCTION = "process"; CATEGORY = "SeeThrough"
    def process(self, layers_depth, tblr_split=True, use_lama=True):
        full = layers_depth["fullpage"]; R = layers_depth["res"]
        subj = np.any(full < 245, axis=2)
        ys, xs = np.nonzero(subj)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tag2 = {}
        for tag, a, b, depth in TAGS:
            ya, yb = int(y0 + (y1 - y0) * a), int(y0 + (y1 - y0) * b)
            xa, xb = x0, x1
            if tag.endswith("-r"):
                xb = (x0 + x1) // 2
            elif tag.endswith("-l"):
                xa = (x0 + x1) // 2
            crop = full[ya:yb, xa:xb]
            alpha = (subj[ya:yb, xa:xb] * 255).astype(np.uint8)
            rgba = np.dstack([crop, alpha])
            if alpha.any():
                tag2[tag] = {"img": rgba, "xyxy": [xa, ya, xb, yb], "depth_median": depth, "tag": tag}
        return ({"tag2pinfo": tag2, "frame_size": (R, R)}, torch.zeros((1, 64, 64, 3)))


class SeeThrough_SavePSD:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"parts": ("SEETHROUGH_PARTS",), "filename_prefix": ("STRING", {"default": "seethrough"})}}
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("info_file",)
    FUNCTION = "save"; CATEGORY = "SeeThrough"; OUTPUT_NODE = True
    def save(self, parts, filename_prefix="seethrough"):
        return ("",)


class SeeThrough_PartsToLayers:  # identical logic to the real node
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"parts": ("SEETHROUGH_PARTS",)}}
    RETURN_TYPES = ("LAYERS",); RETURN_NAMES = ("layers",)
    FUNCTION = "convert"; CATEGORY = "SeeThrough"
    def convert(self, parts):
        tag2pinfo = parts["tag2pinfo"]; canvas_h, canvas_w = parts["frame_size"]
        sorted_tags = sorted(tag2pinfo.keys(), key=lambda t: tag2pinfo[t].get("depth_median", 1), reverse=True)
        items = []
        for z, tag in enumerate(sorted_tags):
            p = tag2pinfo[tag]; img = p["img"]; xyxy = p["xyxy"]
            items.append({"image": torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0), "type": "raster",
                          "name": tag, "x": int(xyxy[0]), "y": int(xyxy[1]), "z_index": z})
        return ({"version": 1, "layers": items, "canvas": (int(canvas_w), int(canvas_h))},)


NODE_CLASS_MAPPINGS = {c.__name__: c for c in [SeeThrough_LoadLayerDiffModel, SeeThrough_LoadDepthModel,
                       SeeThrough_GenerateLayers, SeeThrough_GenerateDepth, SeeThrough_PostProcess,
                       SeeThrough_SavePSD, SeeThrough_PartsToLayers]}
