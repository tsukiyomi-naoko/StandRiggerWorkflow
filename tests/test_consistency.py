"""Regression tests for sheet consistency fixes:
1. RegionCrop ignores a 'head' mask that covers the whole character (falls back to the head estimate).
2. Connection panels are cropped relative to the character in each generated image, so a zoomed-out
   edit still yields the right body band.
3. Connection prompts all carry the pose lock.
    python tests/test_consistency.py   (needs torch, numpy, Pillow)
"""
import importlib.util, json, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "custom_nodes", "ComfyUI-StandRig-Bridge")
spec = importlib.util.spec_from_file_location("srb", os.path.join(PKG, "__init__.py"), submodule_search_locations=[PKG])
srb = importlib.util.module_from_spec(spec); sys.modules["srb"] = srb; spec.loader.exec_module(srb)
N = sys.modules["srb.standrig_nodes"]; P = sys.modules["srb.prompts"]


def character(W=600, H=1200, scale=1.0, cx=300, top=100):
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    h = 1000 * scale
    d.ellipse([cx - 90 * scale, top, cx + 90 * scale, top + 0.2 * h], fill=(240, 200, 180))          # head
    d.rectangle([cx - 110 * scale, top + 0.2 * h, cx + 110 * scale, top + 0.6 * h], fill=(230, 120, 150))  # body
    d.rectangle([cx - 60 * scale, top + 0.6 * h, cx + 60 * scale, top + h], fill=(120, 90, 80))       # legs
    return im


def t(im):
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0)[None]


# 1. whole-body "head" mask -> head estimate is used
img = character()
whole = np.zeros((1200, 600), np.float32); whole[100:1100, 190:410] = 1
crop, info, _ = N.StandRigRegionCrop().run(t(img), 0.45, 512, "head_from_subject_top", 0.24,
                                          region_mask=torch.from_numpy(whole)[None])
assert info["h"] < 500, f"crop should be head-sized, got {info}"
assert info["y"] < 150 and info["y"] + info["h"] > 250, info  # contains the head (y 100..300)
good = np.zeros((1200, 600), np.float32); good[100:300, 210:390] = 1
_, info2, _ = N.StandRigRegionCrop().run(t(img), 0.45, 512, "head_from_subject_top", 0.24,
                                        region_mask=torch.from_numpy(good)[None])
assert abs(info2["h"] - 200 * 1.45) < 5, info2  # a plausible head mask is still used
print("head crop ok:", info["y"], info["h"], "| real mask:", info2["y"], info2["h"])

# 2. zoomed-out edit: the knees band must still show legs, not the chest
pp = N.StandRigPosePrompts().run("connection_targets", "the girl", 30, 20, 15, 15, 8, 8, "auto", "", "wide shot", "")
prompts, ids, _masks, plan = pp
zoomed = character(scale=0.6, cx=300, top=400)   # model returned the character smaller and lower
gen = [t(zoomed) for _ in ids]
_sheet, _panels, data = N.StandRigReferenceSheet().run(["connections"], [t(img)], gen, ids, [plan], [256], [3],
                                                       [False], ["auto"], [True])
knees = data["images"]["thighs-lower-legs"]
arr = np.asarray(knees)
legs = np.all(np.abs(arr.astype(int) - [120, 90, 80]) < 10, axis=2).mean()
body = np.all(np.abs(arr.astype(int) - [230, 120, 150]) < 10, axis=2).mean()
print(f"knees panel: legs {legs:.2f}, body {body:.2f}")
assert legs > body and legs > 0.05, "knee band must follow the character in the generated image"

# 3. pose lock in every connection prompt
assert all(P.POSE_LOCK in p for p in prompts) and len(prompts) == 6
assert not any("Raise" in p or "Bend" in p for p in prompts)
print("ALL CONSISTENCY TESTS PASSED")
