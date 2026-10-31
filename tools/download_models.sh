#!/usr/bin/env bash
# Download every model the StandRig workflows use into a ComfyUI install.
#   bash tools/download_models.sh /path/to/ComfyUI
# Optional: HF_TOKEN=hf_xxx for faster/authenticated Hugging Face downloads.
# See-through models are fetched automatically by ComfyUI-See-through on first run.
set -euo pipefail
COMFY="${1:?usage: download_models.sh /path/to/ComfyUI}"
M="$COMFY/models"
get() {  # dir url
  mkdir -p "$M/$1"; local f="$M/$1/$(basename "$2")"
  if [ -s "$f" ]; then echo "ok   $f"; return; fi
  echo "get  $f"
  curl -L --fail --retry 4 -C - ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} -o "$f.part" "$2" && mv "$f.part" "$f"
}
HF=https://huggingface.co
get diffusion_models   $HF/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors
get text_encoders      $HF/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
get vae                $HF/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
get loras              $HF/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
get loras              $HF/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors
get checkpoints        $HF/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors
get background_removal $HF/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors
echo "done. Low VRAM? also: get diffusion_models $HF/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors"
