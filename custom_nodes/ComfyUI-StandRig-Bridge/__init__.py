"""ComfyUI-StandRig-Bridge: turn one 2D character picture into StandRig inputs
(layered PSD, face/body/connection target sheets, expression variants and a
standrig-visual-reference-v1 manifest)."""
from .standrig_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = None
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
