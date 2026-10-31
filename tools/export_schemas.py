#!/usr/bin/env python3
"""Export the node schemas used by build_workflows.py from a running ComfyUI.

    python tools/export_schemas.py http://127.0.0.1:8188 > tools/node_schemas.json

Model-file combo lists are cleared so the file does not depend on what is
installed on the exporting machine.
"""
import json
import sys
import urllib.request

KEEP_PREFIXES = ("StandRig", "SeeThrough_")
KEEP = {
    "LoadImage", "PreviewImage", "SaveImage", "MarkdownNote", "LoadBackgroundRemovalModel", "RemoveBackground",
    "UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "ModelSamplingAuraFlow", "CFGNorm",
    "CheckpointLoaderSimple", "CLIPTextEncode", "SAM3_Detect", "FluxKontextImageScale", "TextEncodeQwenImageEditPlus",
    "FluxKontextMultiReferenceLatentMethod", "VAEEncode", "VAEDecode", "KSampler", "ComfySwitchNode",
    "PrimitiveStringMultiline", "JoinImageWithAlpha", "ImageCompositor",
}
MODEL_INPUTS = {"unet_name", "clip_name", "vae_name", "lora_name", "ckpt_name", "bg_removal_name", "image"}

url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8188").rstrip("/")
info = json.load(urllib.request.urlopen(url + "/object_info"))
out = {}
for k, v in sorted(info.items()):
    if k in KEEP or k.startswith(KEEP_PREFIXES):
        for sec in ("required", "optional"):
            for name, spec in (v["input"].get(sec) or {}).items():
                if name in MODEL_INPUTS and isinstance(spec[0], list) and k != "SeeThrough_LoadLayerDiffModel":
                    spec[0] = []
        out[k] = v
json.dump(out, sys.stdout, indent=1, sort_keys=True)
