# ComfyUI → StandRig: one picture to a 2.5D rig kit

These ComfyUI workflows take **one static 2D character picture** and produce everything
[StandRig](https://github.com/sayaka-aiart/StandRig) needs before an AI client can rig a 2.5D (Live2D-style) model:

| Output | Why StandRig needs it |
| --- | --- |
| `character.psd`: parts-separated layers (front/back hair, face, eye whites, irises, lashes, brows, mouth, neck, clothes…) with **hidden areas painted in** | StandRig only imports parts-separated PSDs; no automatic part separation |
| Hidden `Expressions/*` layers: blink, smile-eyes, A/I/U/E/O mouths, angry/sad brows | Swap art for StandRig expression and viseme presets (opacity keys) |
| `face-target.png`: neutral, yaw±, pitch±, roll± | Required by the modeling contract (AGENTS.md) before the first modeling write |
| `body-target.png`: neutral, body yaw±, pitch±, roll± | Required |
| `connection-target.png`: hair roots/back hair, jaw/neck, neck/collar/shoulders, sleeves/arms, waist/skirt/thighs, thighs/lower legs | Required |
| `reference-manifest.json` (`standrig-visual-reference-v1`): sha256 of every sheet, panel and the PSD; panel order, parameter extremes, prompts, acceptance checks | Required; modeling transactions are blocked without it |
| `turnaround-sheet.png`: 8 views | Orientation reference for back hair, accessories and depth |

```
picture ─► BiRefNet cutout ─► (optional) Qwen-Edit "normalize to front A-pose"
            │
            ├─► SAM3 "head" ─► face close-up ─┬─► Qwen-Image-Edit-2511 ×4  ─► face-target.png    (roll = exact rotation)
            │                                 └─► Qwen-Image-Edit-2511 ×9  ─► expressions ─► SAM3 masks ─► hidden PSD layers
            ├─► Qwen-Image-Edit-2511 ×6  ─► body-target.png
            ├─► Qwen-Image-Edit-2511 ×6  ─► connection-target.png (cropped to body bands)
            ├─► Qwen-Edit + Multiple-Angles LoRA ×7 ─► turnaround-sheet.png
            └─► See-through (LayerDiff 3D + Marigold depth) ─► LAYERS ─► StandRig PSD writer ─► character.psd
                                                                                    └─► reference pack + manifest
```

## Easy install (Windows, ComfyUI Desktop): StandRig Setup

1. Download this repository (**Code → Download ZIP**) and unzip it anywhere.
2. Double-click **`StandRigSetup.bat`**. It uses ComfyUI Desktop's own Python, so nothing else needs installing.
   Alternatively, get `StandRigSetup.exe` from the [Releases](https://github.com/tsukiyomi-naoko/StandRiggerWorkflow/releases) page.
3. Check what the tool detected at the top. It reads Comfy Desktop's own settings (`%APPDATA%\Comfy Desktop`), so it
   works with both the new multi-install Desktop and the older single-folder one:
   * **ComfyUI install**: pick which install to set up if you have several.
   * **Custom nodes →** where plugins go: the install's folder, or `Documents\ComfyUI` for installs adopted from the old Desktop.
   * **Models download to →** Desktop's DOWNLOADS folder (by default `%LOCALAPPDATA%\Comfy-Desktop\ComfyUI-Shared\models`).
     Models already in any of your model folders are recognised and not downloaded again.
   Use **Browse…** only to point at a different ComfyUI by hand.
   **Close ComfyUI**, then click **Install / update everything**.
4. Restart ComfyUI and open the Workflows sidebar (**W**) → **StandRig** → `standrig_full_pipeline`.

What the tool does:

| Tab | |
| --- | --- |
| **Quick setup** | One click: installs StandRig Bridge and ComfyUI-See-through (and their Python packages, through ComfyUI Desktop's bundled `uv`), copies the workflows into the Workflows sidebar, and downloads all models. Lets you choose the INT8 (smaller) or BF16 Qwen weights; the workflows are switched to match. |
| **Plugins** | Install, update (with update check), or remove any plugin from the catalog. Paste any GitHub URL to add other custom nodes. Old versions are kept in `<ComfyUI>\standrig_backups`. |
| **Models** | Real download sizes, per-model status, resumable downloads with progress and ETA. An optional Hugging Face token can be set in Settings. |
| **Settings** | Tokens, folder overrides, and **Check for tool update** (the tool updates itself). |

The plugin and model lists come from [`installer/catalog.json`](installer/catalog.json) online, so new plugins, models
and tool versions appear without reinstalling anything. A command-line mode exists too: `python installer/standrig_setup.py --help`.

## Workflows (`workflows/`)

| File | What it does | Extra node pack |
| --- | --- | --- |
| `standrig_full_pipeline.json` | Everything above, in 10 labelled groups | ComfyUI-See-through |
| `standrig_reference_targets.json` | Only the StandRig visual gate: face, body and connection sheets + manifest | none |
| `standrig_layers_psd_seethrough.json` | Layered PSD + expression/viseme variant layers | ComfyUI-See-through |
| `standrig_layers_psd_sam3.json` | Quick layered PSD from core SAM3 masks (does **not** paint hidden areas) | none |
| `standrig_turnaround_sheet.json` | 8-view turnaround with the Multiple-Angles LoRA | none |

`workflows/api/*.api.json` are the same graphs in API format, for scripting and `POST /prompt`.

## Manual setup

1. **ComfyUI**: a current build (tested against `comfyanonymous/ComfyUI` @ `6ea48ff`, 2026-10-31). The workflows use these
   core nodes: `TextEncodeQwenImageEditPlus`, `SAM3_Detect`, `RemoveBackground`, `ComfySwitchNode`, `LAYERS`.
2. **This node pack**: copy or symlink `custom_nodes/ComfyUI-StandRig-Bridge` into `ComfyUI/custom_nodes/`. No extra
   pip dependencies.
3. **See-through** (for the PSD workflows):
   ```bash
   cd ComfyUI/custom_nodes && git clone https://github.com/jtydhr88/ComfyUI-See-through && pip install -r ComfyUI-See-through/requirements.txt
   ```
   Its models download from Hugging Face on first run (about 16 GB VRAM at 1280 px; nf4 or `group_offload` for 8–12 GB).
4. **Models** (tens of GB; the StandRig Setup *Models* tab shows exact sizes):
   ```bash
   bash tools/download_models.sh /path/to/ComfyUI
   ```

   | File | Folder |
   | --- | --- |
   | `qwen_image_edit_2511_bf16.safetensors` | `models/diffusion_models` |
   | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `models/text_encoders` |
   | `qwen_image_vae.safetensors` | `models/vae` |
   | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | `models/loras` |
   | `qwen-image-edit-2511-multiple-angles-lora.safetensors` | `models/loras` |
   | `sam3.1_multiplex_fp16.safetensors` | `models/checkpoints` |
   | `birefnet.safetensors` | `models/background_removal` |

   The workflow files also embed these URLs, so ComfyUI's missing-model dialog can download them.
   With 24 GB VRAM or less, set `UNETLoader → weight_dtype` to `fp8_e4m3fn`, or swap in `qwen_image_edit_2511_int8_convrot.safetensors`.

## Run it

1. Drag `workflows/standrig_full_pipeline.json` into ComfyUI and pick your picture in **Load character picture**.
   A clean, full-body, front-facing picture with nothing covering the face gives the best result.
2. Set `project` (output folder name) in **StandRig Layers To PSD** and **StandRig Save Reference Pack**. In each
   **Prompts** node, set `subject` to something like *"the girl with silver twin tails and a red ribbon"*. It helps identity.
3. If the picture is a dynamic pose or a 3/4 view, flip **Normalize pose? (switch)** to `true`.
4. **Run**. Outputs go to `ComfyUI/output/standrig/<project>/`:
   ```
   character.psd  character_psd_preview.png  character_layers.json
   references/  face-target.png body-target.png connection-target.png expression-sheet.png turnaround-sheet.png
                source.png  panels/<sheet>/<panel>.png  reference-manifest.json
   ```
5. **Review every sheet.** The manifest is written as `GENERATED_PENDING_VISUAL_REVIEW`. Re-run a single panel by
   changing the seed of that group's KSampler, or edit the prompt in its **Prompts** node (`extra` is appended to every
   prompt). Bad panels are the main failure mode; see "Quality tips".

### Hand off to StandRig

1. Start StandRig (`npm ci && npm run build && npm start`), open http://127.0.0.1:5180/, then **「PSDを選択」 → 「PSDを読み込む」**
   with `character.psd`. Compare the import with `character_psd_preview.png`.
2. Pin the manifest to the imported model revision:
   ```bash
   python tools/pin_manifest.py ComfyUI/output/standrig/<project>/references/reference-manifest.json
   ```
3. Copy the `references/` folder into your StandRig project folder, outside `workspace/`.
4. Connect your MCP client (see the StandRig README) and start with something like:
   > Read standrig://docs/contract and standrig://docs/guide. The visual references and manifest are in `<path>/references`
   > (face-target.png, body-target.png, connection-target.png, reference-manifest.json; each panel's direction is in
   > `panels[].direction`). Inspect the parts with standrig_context, then set up head yaw/pitch/roll to match the face
   > target extremes. Use the hidden `Expressions/*` layers for blink and the A/I/U/E/O visemes. Dry-run, run numeric QA,
   > commit, then compare actual max-pose renders against the targets.

## Nodes in `ComfyUI-StandRig-Bridge`

| Node | Purpose |
| --- | --- |
| **StandRig Prepare Canvas** | Crop to the character, scale, place on a flat background. Becomes the shared "source canvas" |
| **StandRig Region Crop** | Square close-up around a mask (SAM3 "head") plus `crop_info` for pasting back |
| **StandRig Pose / Expression Prompts** | Prompt lists for face / body / connection / expressions / turnaround presets, with parameter extremes and viewer-side directions |
| **StandRig Reference Sheet** | Builds a sheet in contract panel order; neutral = approved source, roll = exact rotation about the chin or hip pivot; crops connection bands |
| **StandRig Variant Layers** | Masks each expression edit (SAM3 region, or pixel difference) and maps it back onto the source canvas as hidden LAYERS |
| **StandRig Masks To Layers** | SAM3-only part separation, for when See-through isn't available |
| **StandRig String List** | Multiline text to a list (SAM3 prompts) |
| **StandRig Layers To PSD** | Core `LAYERS` to a layered PSD: undoes See-through's square padding, renames tags (L/R = character's side), folders, hidden layers, RLE. Checked against ag-psd 31.0.2, StandRig's parser |
| **StandRig Save Reference Pack + Manifest** | Sheets, panels, `standrig-visual-reference-v1` manifest with hashes |
| **StandRig Pin Manifest** | Fills revision/semanticHash from StandRig's `/api/context` |

Any node that outputs `LAYERS` can feed the PSD writer, including the core **Create Layered Image** editor. You can
nudge See-through layers there before writing the PSD.

## Quality tips

* **Identity drift** is the usual failure. Keep `subject` descriptive, keep Lightning at 4 steps / cfg 1, or bypass the
  Lightning LoRA and use 40 steps / cfg 4 for harder angles.
* **Roll** panels are exact rotations of your approved art (no drift). Body roll defaults to an AI edit because the legs
  have to stay planted. Set `roll_mode` to change either.
* **Default extremes**: head yaw ±30°, pitch ±20°, roll ±15°; body yaw ±15°, pitch ±8°, roll ±8°. They're written into
  the manifest as `ParamAngleX/Y/Z` and `ParamBodyAngleX/Y/Z`. "min"/"max" directions are given from the viewer's side
  in each panel's `direction`. Confirm the sign convention in StandRig; its guide warns not to infer it from parameter names.
* **See-through** handles occlusion completion (face under bangs, back hair, eye whites). Use the connection sheet as a
  painting guide for anything it misses, then repaint in Photoshop/Krita and re-save. The PSD is plain RGB layers.
* Expression layers need the face close-up to stay pixel-aligned with the source. If a variant shifts, raise `grow_px`
  or switch `mask_mode` to `sam_and_difference`.

## What this does not do

* It doesn't rig. StandRig rigs through an MCP AI client, following its contract; these workflows produce that client's inputs.
* Generated sheets are drafts. StandRig's contract requires a human or AI visual review before freezing a model.
* The tests below run on CPU with stand-ins for the GPU models. The real models were not run in this environment.

## Development

```
custom_nodes/ComfyUI-StandRig-Bridge/  node pack (standrig_nodes.py, prompts.py, psd_writer.py)
tools/build_workflows.py               generates every workflow (UI + API) from tools/node_schemas.json
tools/export_schemas.py                refreshes node_schemas.json from a running ComfyUI
tests/                                 PSD round-trip (psd-tools + ag-psd), CPU end-to-end run with stubs,
                                       server-side validation of the real graphs, frontend load check (Playwright)
```

Edit `build_workflows.py` or `prompts.py`, then run `python tools/build_workflows.py`. To run the tests, start ComfyUI on
:8188 with the node pack plus `tests/stubs/*` linked into `custom_nodes`, and placeholder model files present; then run
`COMFY=… AGPSD_DIR=… bash tests/run_all.sh`. Note that `run_all.sh` queues the real workflows to validate them; with real
model files present they will actually execute.

Credits: [StandRig](https://github.com/sayaka-aiart/StandRig), [See-through](https://github.com/shitagaki-lab/see-through)
/ [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through), Qwen-Image-Edit-2511, lightx2v Lightning LoRA,
fal Multiple-Angles LoRA, Meta SAM 3.1, BiRefNet. Check each model's license before commercial use.
