#!/usr/bin/env python3
"""Generate the StandRig ComfyUI workflows (UI + API format).

Usage:
    python tools/build_workflows.py [--schemas tools/node_schemas.json] [--out workflows]

The node schemas file is a trimmed copy of ComfyUI's /object_info (see
tools/export_schemas.py). Every input order, widget value list and output slot
is derived from it, so the files load in the ComfyUI frontend and pass server
validation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "ComfyUI-StandRig-Bridge"))
import prompts as P  # noqa: E402

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
UI_ONLY = {"MarkdownNote", "Note"}

# --------------------------------------------------------------------- models
MODELS = {
    "unet": "qwen_image_edit_2511_int8_convrot.safetensors",
    "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": "qwen_image_vae.safetensors",
    "lightning": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "angles": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
    "sam3": "sam3.1_multiplex_fp16.safetensors",
    "birefnet": "birefnet.safetensors",
}
MODEL_URLS = {
    "qwen_image_edit_2511_int8_convrot.safetensors": ("diffusion_models", "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors"),
    "qwen_image_edit_2511_bf16.safetensors": ("diffusion_models", "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"),
    "qwen_2.5_vl_7b_fp8_scaled.safetensors": ("text_encoders", "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    "qwen_image_vae.safetensors": ("vae", "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"),
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors": ("loras", "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
    "qwen-image-edit-2511-multiple-angles-lora.safetensors": ("loras", "https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors"),
    "sam3.1_multiplex_fp16.safetensors": ("checkpoints", "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"),
    "birefnet.safetensors": ("background_removal", "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors"),
}

COLORS = {"input": "#3f789e", "models": "#444", "face": "#8A8", "body": "#88A", "conn": "#a1309b",
          "expr": "#b58b2a", "turn": "#335", "layers": "#b06634", "save": "#3f5159"}


# ---------------------------------------------------------------------- graph

class Node:
    def __init__(self, g: "Graph", nid: int, ntype: str, widgets: Dict[str, Any], title: Optional[str],
                 group: Optional[str], col: int, mode: int = 0, color: Optional[str] = None):
        self.g, self.id, self.type, self.widgets, self.title = g, nid, ntype, dict(widgets), title
        self.group, self.col, self.mode, self.color = group, col, mode, color
        self.links_in: Dict[str, Tuple["Node", int]] = {}
        self.pos = [0, 0]
        self.size = [340, 100]

    def out(self, key=0) -> Tuple["Node", int]:
        if isinstance(key, int):
            return (self, key)
        names = self.g.output_names(self.type)
        return (self, names.index(key))

    def __getitem__(self, key):
        return self.out(key)


class Graph:
    def __init__(self, schemas: Dict[str, Any]):
        self.s = schemas
        self.nodes: List[Node] = []
        self.groups: List[Tuple[str, str, str]] = []  # key, title, color
        self.notes: List[Tuple[str, str, int]] = []

    # schema helpers
    def schema(self, t: str) -> Dict[str, Any]:
        if t not in self.s:
            raise KeyError(f"node type {t!r} not in schemas")
        return self.s[t]

    def output_names(self, t: str) -> List[str]:
        sc = self.schema(t)
        return list(sc.get("output_name") or sc["output"])

    def inputs(self, t: str) -> List[Tuple[str, Any, Dict[str, Any], bool]]:
        sc = self.schema(t)
        res = []
        for sec in ("required", "optional"):
            order = (sc.get("input_order") or {}).get(sec) or list((sc["input"].get(sec) or {}).keys())
            for name in order:
                spec = sc["input"][sec][name]
                typ = spec[0]
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                res.append((name, typ, opts, sec == "required"))
        return res

    @staticmethod
    def is_widget(typ, opts) -> bool:
        if opts.get("forceInput"):
            return False
        return isinstance(typ, list) or typ in WIDGET_TYPES

    def group(self, key: str, title: str, color: str):
        self.groups.append((key, title, color))

    def add(self, t: str, widgets: Optional[Dict[str, Any]] = None, inputs: Optional[Dict[str, Tuple[Node, int]]] = None,
            title: Optional[str] = None, group: Optional[str] = None, col: int = 0, mode: int = 0) -> Node:
        if t not in UI_ONLY:
            self.schema(t)
            known = {i[0] for i in self.inputs(t)}
            for k in list((widgets or {}).keys()) + list((inputs or {}).keys()):
                if k not in known:
                    raise KeyError(f"{t} has no input {k!r}")
        n = Node(self, len(self.nodes) + 1, t, widgets or {}, title, group, col, mode)
        for k, v in (inputs or {}).items():
            n.links_in[k] = v
        self.nodes.append(n)
        return n

    def note(self, group: str, text: str, col: int = 0) -> Node:
        return self.add("MarkdownNote", {"text": text}, group=group, col=col)

    # ------------------------------------------------------------- layout
    def _estimate_size(self, n: Node) -> List[int]:
        if n.type in UI_ONLY:
            lines = n.widgets["text"].count("\n") + 1 + len(n.widgets["text"]) // 60
            return [420, min(900, 60 + 18 * lines)]
        h = 46
        for name, typ, opts, _ in self.inputs(n.type):
            if self.is_widget(typ, opts):
                h += 24
                if typ == "STRING" and opts.get("multiline"):
                    h += 76
            else:
                h += 22
        h += 22 * max(0, len(self.output_names(n.type)) - 1)
        if n.type in ("PreviewImage", "SaveImage"):
            h = max(h, 300)
        return [360 if n.type.startswith("StandRig") or n.type.startswith("TextEncode") else 320, h]

    def layout(self):
        y = 0
        gap_x, gap_y, pad = 40, 24, 40
        self.bounds = {}

        def num(g):  # lay groups out in their numbered order ("1 · …", "2 · …", …)
            m = re.match(r"\s*(\d+)", g[1])
            return int(m.group(1)) if m else 999
        self.groups.sort(key=num)
        for key, title, color in self.groups:
            members = [n for n in self.nodes if n.group == key]
            if not members:
                continue
            cols: Dict[int, List[Node]] = {}
            for n in members:
                cols.setdefault(n.col, []).append(n)
            x = pad
            max_h = 0
            for c in sorted(cols):
                cy = y + pad + 40
                w = 0
                for n in cols[c]:
                    n.size = self._estimate_size(n)
                    n.pos = [x, cy]
                    cy += n.size[1] + gap_y
                    w = max(w, n.size[0])
                max_h = max(max_h, cy - y)
                x += w + gap_x
            self.bounds[key] = [0, y, x + pad, max_h + pad]
            y += max_h + pad + 80

    # ------------------------------------------------------------- export
    def widget_values(self, n: Node) -> List[Any]:
        if n.type in UI_ONLY:
            return [n.widgets["text"]]
        vals = []
        for name, typ, opts, _ in self.inputs(n.type):
            if not self.is_widget(typ, opts):
                continue
            if name in n.widgets:
                v = n.widgets[name]
            elif "default" in opts:
                v = opts["default"]
            elif isinstance(typ, list) and typ:
                v = typ[0]
            elif typ == "COMBO" and opts.get("options"):
                v = opts["options"][0]
            else:
                v = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(typ, None)
            vals.append(v)
            if typ == "INT" and (opts.get("control_after_generate") or name in ("seed", "noise_seed")):
                vals.append(n.widgets.get(f"{name}__control", "fixed"))
            if opts.get("image_upload"):
                vals.append("image")
        return vals

    def to_ui(self) -> Dict[str, Any]:
        self.layout()
        links = []
        link_id = 0
        ui_nodes = {}
        for n in self.nodes:
            if n.type in UI_ONLY:
                ui_nodes[n.id] = {"id": n.id, "type": n.type, "pos": n.pos, "size": n.size, "flags": {},
                                  "order": 0, "mode": 0, "inputs": [], "outputs": [],
                                  "properties": {}, "widgets_values": self.widget_values(n),
                                  "color": "#432", "bgcolor": "#653"}
                continue
            ins = []
            for name, typ, opts, _req in self.inputs(n.type):
                widget = self.is_widget(typ, opts)
                if widget and name not in n.links_in:
                    continue
                t = "COMBO" if isinstance(typ, list) else typ
                entry = {"name": name, "type": t, "link": None}
                if widget:
                    entry["widget"] = {"name": name}
                ins.append(entry)
            outs = []
            sc = self.schema(n.type)
            out_is_list = sc.get("output_is_list") or [False] * len(sc["output"])
            for i, (t, nm) in enumerate(zip(sc["output"], self.output_names(n.type))):
                o = {"name": nm, "type": t if not isinstance(t, list) else "COMBO", "links": [], "slot_index": i}
                if out_is_list[i]:
                    o["shape"] = 6
                outs.append(o)
            props = {"Node name for S&R": n.type}
            if n.type in ("UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "CheckpointLoaderSimple",
                          "LoadBackgroundRemovalModel"):
                fn = next((v for v in n.widgets.values() if isinstance(v, str) and v in MODEL_URLS), None)
                if fn:
                    props["models"] = [{"name": fn, "url": MODEL_URLS[fn][1], "directory": MODEL_URLS[fn][0]}]
            d = {"id": n.id, "type": n.type, "pos": n.pos, "size": n.size, "flags": {}, "order": 0,
                 "mode": n.mode, "inputs": ins, "outputs": outs, "properties": props,
                 "widgets_values": self.widget_values(n)}
            if n.title:
                d["title"] = n.title
            ui_nodes[n.id] = d
        # links
        for n in self.nodes:
            for name, (src, slot) in n.links_in.items():
                link_id += 1
                tgt = ui_nodes[n.id]
                ti = next(i for i, e in enumerate(tgt["inputs"]) if e["name"] == name)
                st = ui_nodes[src.id]["outputs"][slot]
                tgt["inputs"][ti]["link"] = link_id
                st["links"].append(link_id)
                links.append([link_id, src.id, slot, n.id, ti, st["type"]])
        # execution order (topological)
        order, seen = {}, set()

        def visit(nn: Node):
            if nn.id in seen:
                return
            seen.add(nn.id)
            for s, _ in nn.links_in.values():
                visit(s)
            order[nn.id] = len(order)
        for nn in self.nodes:
            visit(nn)
        for nid, d in ui_nodes.items():
            d["order"] = order.get(nid, 0)
        groups = []
        for key, title, color in self.groups:
            if key in self.bounds:
                groups.append({"title": title, "bounding": self.bounds[key], "color": color, "font_size": 24,
                               "flags": {}})
        models = []
        for n in self.nodes:
            for v in n.widgets.values():
                if isinstance(v, str) and v in MODEL_URLS and not any(m["name"] == v for m in models):
                    models.append({"name": v, "url": MODEL_URLS[v][1], "directory": MODEL_URLS[v][0]})
        return {"last_node_id": len(self.nodes), "last_link_id": link_id, "nodes": list(ui_nodes.values()),
                "links": links, "groups": groups, "config": {}, "extra": {"ds": {"scale": 0.45, "offset": [0, 0]}},
                "models": models, "version": 0.4}

    def to_api(self) -> Dict[str, Any]:
        api = {}
        for n in self.nodes:
            if n.type in UI_ONLY or n.mode == 4:
                continue
            inputs = {}
            for name, typ, opts, _req in self.inputs(n.type):
                if name in n.links_in:
                    src, slot = n.links_in[name]
                    inputs[name] = [str(src.id), slot]
                elif self.is_widget(typ, opts):
                    if name in n.widgets:
                        inputs[name] = n.widgets[name]
                    elif "default" in opts:
                        inputs[name] = opts["default"]
                    elif isinstance(typ, list) and typ:
                        inputs[name] = typ[0]
                    elif typ == "COMBO" and opts.get("options"):
                        inputs[name] = opts["options"][0]
            api[str(n.id)] = {"class_type": n.type, "inputs": inputs, "_meta": {"title": n.title or n.type}}
        return api


# ------------------------------------------------------------------ builders

class Ctx:
    """Shared nodes for one workflow (models, source image...)."""
    def __init__(self, g: Graph, fake: bool, steps: int, cfg: float, seed: int):
        self.g, self.fake, self.steps, self.cfg, self.seed = g, fake, steps, cfg, seed
        self.model_edit = self.model_angles = self.clip = self.vae = None
        self.sam_model = self.sam_clip = None


def build_input(c: Ctx, image_name: str, normalize: bool):
    g = c.g
    g.group("input", "1 · Input picture → clean canvas", COLORS["input"])
    g.note("input", (
        "## 1 · Input\n"
        "Load **one** picture of your character (full body, front facing works best).\n\n"
        "* Background is removed with BiRefNet and the character is centred on a white canvas.\n"
        "* `Normalize pose` (optional, off by default): Qwen-Image-Edit redraws the character front-facing, "
        "eyes open, mouth closed, arms slightly away from the body — the pose Live2D/StandRig rigs need. "
        "Turn it on when your picture is a dynamic pose or 3/4 view.\n"
        "* Everything downstream uses the **source canvas** from `StandRig Prepare Canvas`."), col=0)
    load = g.add("LoadImage", {"image": image_name}, group="input", col=0, title="Load character picture")
    if c.fake:
        prep = g.add("StandRigPrepareCanvas", {"long_side": 1024}, {"image": load[0]}, group="input", col=1)
    else:
        bgm = g.add("LoadBackgroundRemovalModel", {"bg_removal_name": MODELS["birefnet"]}, group="input", col=1)
        rb = g.add("RemoveBackground", {}, {"bg_removal_model": bgm[0], "image": load[0]}, group="input", col=1)
        prep = g.add("StandRigPrepareCanvas", {"long_side": 1536}, {"image": load[0], "foreground_mask": rb[0]},
                     group="input", col=1)
    src = prep[0]
    if normalize:
        pr = g.add("PrimitiveStringMultiline", {"value": P.NORMALIZE_PROMPT.replace("{subject}", "the character")},
                   group="input", col=2, title="Normalize-pose prompt")
        dec = edit_chain(c, "input", prep[0], pr[0], c.model_edit, col=2, label="normalize")
        sw = g.add("ComfySwitchNode", {"switch": False}, {"on_false": prep[0], "on_true": dec[0]}, group="input",
                   col=3, title="Normalize pose? (switch)")
        prep2 = g.add("StandRigPrepareCanvas", {"long_side": 1536}, {"image": sw[0]}, group="input", col=3,
                      title="Source canvas")
        src = prep2[0]
    g.add("PreviewImage", {}, {"images": src}, group="input", col=4, title="Source canvas")
    return src, prep


def build_models(c: Ctx, need_angles: bool):
    g = c.g
    g.group("models", "2 · Models (Qwen-Image-Edit-2511 + Lightning 4-step)", COLORS["models"])
    g.note("models", (
        "## 2 · Models\n"
        "Qwen-Image-Edit-2511 + Lightning LoRA (4 steps, cfg 1). For maximum quality bypass the Lightning LoRA "
        "and set every KSampler to **steps 40 / cfg 4**.\n\n"
        "Uses the INT8 ConvRot weights (19 GB). Best quality on a 32 GB+ GPU: pick `qwen_image_edit_2511_bf16.safetensors` "
        "in the UNETLoader (38 GB; StandRig Setup can download and switch it).\n\n"
        "The Multiple-Angles LoRA (fal) is only used by the turnaround sheet; StandRig targets use exact-degree "
        "instructions on the base model."), col=0)
    if c.fake:
        return
    unet = g.add("UNETLoader", {"unet_name": MODELS["unet"], "weight_dtype": "default"}, group="models", col=1)
    c.clip = g.add("CLIPLoader", {"clip_name": MODELS["clip"], "type": "qwen_image", "device": "default"},
                   group="models", col=1)[0]
    c.vae = g.add("VAELoader", {"vae_name": MODELS["vae"]}, group="models", col=1)[0]
    ms = g.add("ModelSamplingAuraFlow", {"shift": 3.1}, {"model": unet[0]}, group="models", col=2)
    cn = g.add("CFGNorm", {"strength": 1.0}, {"model": ms[0]}, group="models", col=2)
    lt = g.add("LoraLoaderModelOnly", {"lora_name": MODELS["lightning"], "strength_model": 1.0}, {"model": cn[0]},
               group="models", col=2, title="Lightning 4-step LoRA")
    c.model_edit = lt[0]
    if need_angles:
        an = g.add("LoraLoaderModelOnly", {"lora_name": MODELS["angles"], "strength_model": 0.9}, {"model": unet[0]},
                   group="models", col=3, title="Multiple-Angles LoRA (turnaround)")
        ms2 = g.add("ModelSamplingAuraFlow", {"shift": 3.1}, {"model": an[0]}, group="models", col=3)
        cn2 = g.add("CFGNorm", {"strength": 1.0}, {"model": ms2[0]}, group="models", col=3)
        lt2 = g.add("LoraLoaderModelOnly", {"lora_name": MODELS["lightning"], "strength_model": 1.0},
                    {"model": cn2[0]}, group="models", col=3, title="Lightning 4-step LoRA")
        c.model_angles = lt2[0]


def build_sam(c: Ctx):
    if c.fake or c.sam_model:
        return
    g = c.g
    ck = g.add("CheckpointLoaderSimple", {"ckpt_name": MODELS["sam3"]}, group="models", col=4, title="SAM3.1 (segmentation)")
    c.sam_model, c.sam_clip = ck[0], ck[1]


def sam_mask(c: Ctx, group: str, image, text_or_link, col: int, threshold: float = 0.4):
    g = c.g
    if c.fake:
        return g.add("StandRigTestFakeRegionMask", {} if not isinstance(text_or_link, str) else {"text": text_or_link},
                     {"image": image, **({} if isinstance(text_or_link, str) else {"text": text_or_link})},
                     group=group, col=col)[0]
    enc = g.add("CLIPTextEncode", {} if not isinstance(text_or_link, str) else {"text": text_or_link},
                {"clip": c.sam_clip, **({} if isinstance(text_or_link, str) else {"text": text_or_link})},
                group=group, col=col, title="SAM3 text prompt")
    det = g.add("SAM3_Detect", {"threshold": threshold, "refine_iterations": 2, "individual_masks": False},
                {"model": c.sam_model, "image": image, "conditioning": enc[0]}, group=group, col=col)
    return det[0]


def edit_chain(c: Ctx, group: str, image, prompt_link, model, col: int, label: str = ""):
    """Qwen-Image-Edit-2511 instruction edit. Runs once per prompt when prompt_link is a list."""
    g = c.g
    if c.fake:
        return g.add("StandRigTestFakeEdit", {}, {"image": image, "prompt": prompt_link}, group=group, col=col,
                     title=f"(test) fake edit {label}")
    sc = g.add("FluxKontextImageScale", {}, {"image": image}, group=group, col=col)
    pos = g.add("TextEncodeQwenImageEditPlus", {}, {"clip": c.clip, "prompt": prompt_link, "vae": c.vae,
                                                     "image1": sc[0]}, group=group, col=col,
                title=f"Edit prompt {label}".strip())
    neg = g.add("TextEncodeQwenImageEditPlus", {"prompt": ""}, {"clip": c.clip, "vae": c.vae, "image1": sc[0]},
                group=group, col=col, title="Negative (empty)")
    pm = g.add("FluxKontextMultiReferenceLatentMethod", {"reference_latents_method": "index_timestep_zero"},
               {"conditioning": pos[0]}, group=group, col=col + 1)
    nm = g.add("FluxKontextMultiReferenceLatentMethod", {"reference_latents_method": "index_timestep_zero"},
               {"conditioning": neg[0]}, group=group, col=col + 1)
    enc = g.add("VAEEncode", {}, {"pixels": sc[0], "vae": c.vae}, group=group, col=col + 1)
    ks = g.add("KSampler", {"seed": c.seed, "steps": c.steps, "cfg": c.cfg, "sampler_name": "euler",
                            "scheduler": "simple", "denoise": 1.0},
               {"model": model, "positive": pm[0], "negative": nm[0], "latent_image": enc[0]}, group=group,
               col=col + 1)
    return g.add("VAEDecode", {}, {"samples": ks[0], "vae": c.vae}, group=group, col=col + 1)


def prompts_node(c: Ctx, group: str, preset: str, col: int = 0, **w):
    return c.g.add("StandRigPosePrompts", {"preset": preset, **w}, group=group, col=col,
                   title=f"Prompts · {preset}")


def build_face_crop(c: Ctx, src, alpha):
    g = c.g
    g.group("face_crop", "3 · Face close-up (SAM3 'head' → square crop)", COLORS["face"])
    build_sam(c)
    m = sam_mask(c, "face_crop", src, "head", col=0)
    crop = g.add("StandRigRegionCrop", {"padding": 0.45, "size": 1024}, {"image": src, "region_mask": m,
                                                                           "subject_alpha": alpha},
                 group="face_crop", col=1)
    g.add("PreviewImage", {}, {"images": crop[0]}, group="face_crop", col=2, title="Face close-up")
    return crop


def build_sheet(c: Ctx, key: str, title: str, color: str, preset: str, sheet: str, image, model, note: str,
                pw: Optional[Dict[str, Any]] = None, sheet_w: Optional[Dict[str, Any]] = None):
    g = c.g
    g.group(key, title, color)
    g.note(key, note, col=0)
    pr = prompts_node(c, key, preset, col=0, **(pw or {}))
    dec = edit_chain(c, key, image, pr["prompts"], model, col=1, label=f"· {sheet}")
    sh = g.add("StandRigReferenceSheet", {"sheet": sheet, **(sheet_w or {})},
               {"neutral": image, "generated": dec[0], "panel_ids": pr["panel_ids"], "plan_json": pr["plan_json"]},
               group=key, col=3)
    g.add("PreviewImage", {}, {"images": sh[0]}, group=key, col=4, title=f"{sheet} sheet")
    return sh, dec, pr


NOTE_FACE = ("## 4 · Face targets (StandRig contract)\n"
             "Panels: neutral, yaw-min/max, pitch-min/max, roll-min/max on the face close-up.\n\n"
             "* yaw/pitch are AI edits with exact-degree instructions (defaults ±30° / ±20°).\n"
             "* roll is an **exact rotation** of the approved neutral art around the chin/neck pivot, so it never "
             "drifts off-model. Set `roll_mode` = ai to generate it instead.\n"
             "* Directions are written from the viewer's side in the manifest (`direction`).")
NOTE_BODY = ("## 5 · Body targets\n"
             "neutral, body-yaw-min/max, body-pitch-min/max, body-roll-min/max on the full body. Body roll is an "
             "AI edit by default (legs stay planted); set `roll_mode` = rotate for a rigid rotation about the hips.")
NOTE_CONN = ("## 6 · Connection targets\n"
             "Hidden overlap art the PSD needs so nothing gaps at maximum motion: hair roots/back hair, jaw/neck, "
             "neck/collar/shoulders, sleeves/arms, waist/skirt/thighs, thighs/lower legs. Each panel is cropped to "
             "its body band. Use these as painting references for layers See-through could not complete.")
NOTE_EXPR = ("## 7 · Expressions & visemes\n"
             "Blink, smile-eyes, A/I/U/E/O mouths and brows on the face close-up. SAM3 isolates the changed region "
             "(eyes / mouth / eyebrows) and `StandRig Variant Layers` pastes each one back on the source canvas as a "
             "**hidden PSD layer** under `Variants/Expressions/…`, ready for StandRig opacity keys and viseme presets.")
NOTE_TURN = ("## 8 · Turnaround sheet (Multiple-Angles LoRA)\n"
             "8 views around the character with the fal Multiple-Angles LoRA (`<sks> front-left quarter view …`). "
             "Use it as an orientation reference for back hair, accessories and body depth. Not part of the StandRig "
             "gate, but saved into the reference pack.")
NOTE_LAYERS = ("## 9 · Part separation → PSD\n"
               "See-through (SIGGRAPH 2026) decomposes the source canvas into up to 24 semantic layers **with the "
               "hidden parts painted in** (face under the bangs, back hair, eye whites…) and orders them by depth.\n\n"
               "`StandRig Layers To PSD` maps them back to the source canvas, renames tags (hairf → Front Hair, "
               "eyewhite-l → Eye White L; L/R = character's own side), adds the expression variants as hidden layers "
               "and writes `output/standrig/<project>/<name>.psd` with ag-psd-compatible folders.\n\n"
               "Needs the custom node pack **ComfyUI-See-through** (models auto-download from Hugging Face, ~16 GB "
               "VRAM at 1280; use nf4 / group_offload for 8–12 GB).")
NOTE_SAVE = ("## 10 · Save the StandRig reference pack\n"
             "Writes to `output/standrig/<project>/references/`:\n"
             "`face-target.png`, `body-target.png`, `connection-target.png`, `expression-sheet.png`, "
             "`turnaround-sheet.png`, per-panel PNGs and **reference-manifest.json** "
             "(`standrig-visual-reference-v1`, sha256 of every image and of the PSD).\n\n"
             "After importing the PSD in StandRig run `tools/pin_manifest.py` (or the *Pin Manifest* node) to pin "
             "the model revision, then hand the folder to your AI client.")


def build_seethrough(c: Ctx, src, extra_layers, project: str, low: bool = False):
    g = c.g
    g.group("layers", "9 · Part separation (See-through) → StandRig PSD" + (" · low memory" if low else ""),
            COLORS["layers"])
    g.note("layers", NOTE_LAYERS + (NOTE_LOW_ST if low else ""), col=0)
    q = "nf4" if low else "none"
    ld = g.add("SeeThrough_LoadLayerDiffModel", {"model": "layerdifforg/seethroughv0.0.2_layerdiff3d",
                                                "quant_mode": q, "cache_tag_embeds": True,
                                                "group_offload": low, "auto_download": True},
               group="layers", col=1)
    dm = g.add("SeeThrough_LoadDepthModel", {"model": "layerdifforg/seethroughv0.0.1_marigold", "quant_mode": q,
                                            "cache_tag_embeds": True, "group_offload": low, "auto_download": True},
               group="layers", col=1)
    gl = g.add("SeeThrough_GenerateLayers", {"seed": 42, "resolution": 1024 if low else 1280,
                                            "num_inference_steps": 30},
               {"image": src, "layerdiff_model": ld[0]}, group="layers", col=2)
    gd = g.add("SeeThrough_GenerateDepth", {"seed": 42, "resolution_depth": 720 if low else -1},
               {"layers": gl[0], "depth_model": dm[0]}, group="layers", col=2)
    pp = g.add("SeeThrough_PostProcess", {"tblr_split": True, "use_lama": True}, {"layers_depth": gd[0]},
               group="layers", col=2)
    pl = g.add("SeeThrough_PartsToLayers", {}, {"parts": pp[0]}, group="layers", col=3)
    g.add("PreviewImage", {}, {"images": pp[1]}, group="layers", col=3, title="See-through reconstruction")
    ins = {"layers": pl[0], "source_image": src}
    if extra_layers is not None:
        ins["extra_layers"] = extra_layers
    psd = g.add("StandRigLayersToPSD", {"project": project, "filename": "character", "fit_mode": "auto",
                                        "rename_parts": True, "min_alpha_pixels": 16, "add_reference_layer": False},
                ins, group="layers", col=4)
    g.add("PreviewImage", {}, {"images": psd[1]}, group="layers", col=4, title="PSD composite check")
    return psd


NOTE_LOW_ST = ("\n\n**Low-memory settings** (8 GB VRAM / 16 GB RAM): 4-bit nf4 models (smaller download, ~8 GB "
               "VRAM), group offload, 1024 px layers, 720 px depth. Slower, but it fits. The first run downloads the "
               "nf4 models from Hugging Face; that shows as 0 % in the queue, watch the console.")
NOTE_LOW_1 = ("## Low memory · step 1 of 2\n"
              "Run this first. It makes the StandRig reference sheets and the expression/viseme variants with "
              "Qwen-Image-Edit, saves them into `output/standrig/<project>/`, then **frees ComfyUI's memory** when it "
              "finishes.\n\nThen open **standrig_lowmem_2_psd** with the **same project name** and run it: it adds the "
              "See-through layers and writes the PSD. The two big models are never loaded together.\n\n"
              "On 16 GB RAM close other apps (browser, games) while it runs.")
NOTE_LOW_2 = ("## Low memory · step 2 of 2\n"
              "Run **standrig_lowmem_1_sheets** first with the same `project` name. This step loads the exact source "
              "canvas and the expression layers it saved (no Qwen model is loaded here), runs See-through with "
              "low-memory settings and writes `character.psd`, then updates the same reference-manifest.json with "
              "the PSD's hash and frees memory.")


def build_lowmem_step2(schemas, fake: bool = False, project: str = "my_character") -> Graph:
    g = Graph(schemas)
    c = Ctx(g, fake, steps=4, cfg=1.0, seed=20261031)
    g.group("input", "1 · Load what step 1 saved", COLORS["input"])
    g.note("input", NOTE_LOW_2, col=0)
    lp = g.add("StandRigLoadProject", {"project": project, "layers_name": "expressions"}, group="input", col=1,
               title="Project from step 1 (same name!)")
    g.add("PreviewImage", {}, {"images": lp[0]}, group="input", col=2, title="Source canvas from step 1")
    psd = build_seethrough(c, lp[0], lp[1], project, low=True)
    g.group("save", "10 · Update the reference pack + free memory", COLORS["save"])
    pack = g.add("StandRigSaveReferencePack", {"project": project}, {"psd_path": psd[0]}, group="save", col=1)
    g.add("StandRigFreeMemory", {}, {"after": pack[0]}, group="save", col=2)
    return g


def build_sam_layers(c: Ctx, src, alpha, extra_layers, project: str):
    g = c.g
    g.group("layers", "9 · Part separation (SAM3 masks, no inpainting) → StandRig PSD", COLORS["layers"])
    g.note("layers", (
        "## 9 · Quick part separation with SAM3 only\n"
        "No extra node pack needed. Each line of the list is a SAM3 text prompt and becomes one layer, **top-most "
        "first** (earlier lines win overlapping pixels).\n\n"
        "⚠ Hidden areas are NOT painted in (no forehead under the bangs, no neck under the jaw). Paint them using "
        "the connection-target sheet, or use the See-through workflow for automatic completion."), col=0)
    build_sam(c)
    names = g.add("StandRigStringList", {"text": "front hair\neyebrows\neyes\nmouth\nnose\nface\nears\nneck\nhands\n"
                                                 "arms\nupper body clothes\nskirt\nlegs\nshoes\nback hair",
                                         "separator": "newline"}, group="layers", col=1, title="Parts (top → bottom)")
    masks = sam_mask(c, "layers", src, names[0], col=2, threshold=0.35)
    ml = g.add("StandRigMasksToLayers", {"exclusive": True, "add_remainder": True},
               {"image": src, "masks": masks, "names": names[0], "subject_alpha": alpha}, group="layers", col=3)
    ins = {"layers": ml[0], "source_image": src}
    if extra_layers is not None:
        ins["extra_layers"] = extra_layers
    psd = g.add("StandRigLayersToPSD", {"project": project, "filename": "character_sam3", "fit_mode": "none",
                                        "rename_parts": False}, ins, group="layers", col=4)
    g.add("PreviewImage", {}, {"images": psd[1]}, group="layers", col=4, title="PSD composite check")
    return psd


def build_workflow(schemas, kind: str, fake: bool = False, project: str = "my_character",
                   image_name: str = "character.png") -> Graph:
    g = Graph(schemas)
    c = Ctx(g, fake, steps=4, cfg=1.0, seed=20261031)
    want = {
        "full": dict(targets=True, expr=True, turn=True, layers="seethrough", normalize=True),
        "targets": dict(targets=True, expr=False, turn=False, layers=None, normalize=True),
        "psd": dict(targets=False, expr=True, turn=False, layers="seethrough", normalize=True),
        "psd_sam3": dict(targets=False, expr=True, turn=False, layers="sam3", normalize=True),
        "turnaround": dict(targets=False, expr=False, turn=True, layers=None, normalize=False),
        "e2e_test": dict(targets=True, expr=True, turn=True, layers="seethrough", normalize=False),
        "lowmem_1": dict(targets=True, expr=True, turn=False, layers=None, normalize=True, save_layers=True,
                         free=True),
        "e2e_low1": dict(targets=True, expr=True, turn=False, layers=None, normalize=False, save_layers=True,
                         free=True),
    }[kind]
    build_models(c, need_angles=want["turn"])
    src, prep = build_input(c, image_name, want["normalize"] and not fake)
    alpha = prep[1]
    face_sheet = body_sheet = conn_sheet = expr_sheet = turn_sheet = None
    extra = None
    crop = None
    if want["targets"] or want["expr"]:
        crop = build_face_crop(c, src, alpha)
    if want["targets"]:
        face_sheet, _, _ = build_sheet(c, "face", "4 · Face targets → face-target.png", COLORS["face"], "face_targets",
                                       "face", crop[0], c.model_edit, NOTE_FACE)
        body_sheet, _, _ = build_sheet(c, "body", "5 · Body targets → body-target.png", COLORS["body"], "body_targets",
                                       "body", src, c.model_edit, NOTE_BODY)
        conn_sheet, _, _ = build_sheet(c, "conn", "6 · Connection overlap targets → connection-target.png",
                                       COLORS["conn"], "connection_targets", "connections", src, c.model_edit,
                                       NOTE_CONN, sheet_w={"columns": 3})
    if want["expr"]:
        expr_sheet, dec, pr = build_sheet(c, "expr", "7 · Expressions & visemes → hidden PSD layers", COLORS["expr"],
                                          "expressions_visemes", "expressions", crop[0], c.model_edit, NOTE_EXPR,
                                          sheet_w={"columns": 5, "panel_height": 512})
        build_sam(c)
        masks = sam_mask(c, "expr", dec[0], pr["mask_prompts"], col=5)
        vl = g.add("StandRigVariantLayers", {"group": "Expressions", "mask_mode": "sam_or_difference"},
                   {"variants": dec[0], "panel_ids": pr["panel_ids"], "crop_info": crop[1], "region_masks": masks,
                    "neutral_crop": crop[0]}, group="expr", col=6)
        g.add("PreviewImage", {}, {"images": vl[1]}, group="expr", col=6, title="Variant masks")
        extra = vl[0]
    if want["turn"]:
        turn_sheet, _, _ = build_sheet(c, "turn", "8 · Turnaround (Multiple-Angles LoRA)", COLORS["turn"],
                                       "turnaround_multiangle_lora", "turnaround", src, c.model_angles or c.model_edit,
                                       NOTE_TURN, sheet_w={"columns": 4, "panel_height": 640})
    psd = None
    if want["layers"] == "seethrough":
        psd = build_seethrough(c, src, extra, project)
    elif want["layers"] == "sam3":
        psd = build_sam_layers(c, src, alpha, extra, project)
    if any([face_sheet, body_sheet, conn_sheet, expr_sheet, turn_sheet, psd]):
        g.group("save", "10 · StandRig reference pack + manifest", COLORS["save"])
        g.note("save", NOTE_SAVE, col=0)
        ins = {"source_image": src}
        for k, v in (("face_sheet", face_sheet), ("body_sheet", body_sheet), ("connection_sheet", conn_sheet),
                     ("expression_sheet", expr_sheet), ("turnaround_sheet", turn_sheet)):
            if v is not None:
                ins[k] = v[2]
        if psd is not None:
            ins["psd_path"] = psd[0]
        pack = g.add("StandRigSaveReferencePack", {"project": project}, ins, group="save", col=1)
        if want.get("save_layers") and extra is not None:
            g.add("StandRigSaveLayers", {"project": project, "name": "expressions"}, {"layers": extra},
                  group="save", col=1, title="Save expression layers for step 2")
        if want.get("free"):
            g.note("save", NOTE_LOW_1, col=2)
            g.add("StandRigFreeMemory", {}, {"after": pack[0]}, group="save", col=2)
    return g


WORKFLOWS = {
    "full": ("standrig_full_pipeline", "Everything: clean-up → face/body/connection targets → expressions → "
             "turnaround → See-through PSD → reference pack"),
    "targets": ("standrig_reference_targets", "StandRig visual gate only: face, body and connection target sheets + manifest"),
    "psd": ("standrig_layers_psd_seethrough", "Layered PSD with See-through + expression/viseme variant layers"),
    "psd_sam3": ("standrig_layers_psd_sam3", "Layered PSD with core SAM3 masks only (no extra node pack)"),
    "turnaround": ("standrig_turnaround_sheet", "8-view turnaround sheet with the Multiple-Angles LoRA"),
    "lowmem_1": ("standrig_lowmem_1_sheets", "Low memory, step 1 of 2: reference sheets + expression layers, "
                 "then frees memory"),
    "lowmem_2": ("standrig_lowmem_2_psd", "Low memory, step 2 of 2: See-through (nf4, offload) PSD from step 1's "
                 "project"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas", default=os.path.join(HERE, "node_schemas.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "workflows"))
    ap.add_argument("--e2e", help="also write the CPU end-to-end test workflow (API) to this path")
    ap.add_argument("--e2e-image", default="standrig_test_character.png")
    ap.add_argument("--e2e-low", help="also write the two low-memory CPU test workflows (API) with this path prefix")
    args = ap.parse_args()
    schemas = json.load(open(args.schemas, encoding="utf-8"))
    os.makedirs(os.path.join(args.out, "api"), exist_ok=True)
    for kind, (name, desc) in WORKFLOWS.items():
        g = build_lowmem_step2(schemas) if kind == "lowmem_2" else build_workflow(schemas, kind)
        ui = g.to_ui()
        ui["extra"]["standrig"] = {"description": desc}
        with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(ui, f, ensure_ascii=False, indent=1)
        with open(os.path.join(args.out, "api", f"{name}.api.json"), "w", encoding="utf-8") as f:
            json.dump(g.to_api(), f, ensure_ascii=False, indent=1)
        print(f"{name}: {len(ui['nodes'])} nodes, {len(ui['links'])} links")
    if args.e2e:
        g = build_workflow(schemas, "e2e_test", fake=True, project="e2e_test", image_name=args.e2e_image)
        with open(args.e2e, "w", encoding="utf-8") as f:
            json.dump(g.to_api(), f, ensure_ascii=False, indent=1)
        with open(args.e2e.replace(".api.json", ".ui.json"), "w", encoding="utf-8") as f:
            json.dump(g.to_ui(), f, ensure_ascii=False, indent=1)
        print(f"e2e test workflow: {len(g.nodes)} nodes")
    if args.e2e_low:
        g1 = build_workflow(schemas, "e2e_low1", fake=True, project="e2e_low", image_name=args.e2e_image)
        g2 = build_lowmem_step2(schemas, fake=True, project="e2e_low")
        for suffix, gg in (("1", g1), ("2", g2)):
            with open(f"{args.e2e_low}{suffix}.api.json", "w", encoding="utf-8") as f:
                json.dump(gg.to_api(), f, ensure_ascii=False, indent=1)
        print(f"low-memory e2e workflows: {len(g1.nodes)} + {len(g2.nodes)} nodes")


if __name__ == "__main__":
    main()
