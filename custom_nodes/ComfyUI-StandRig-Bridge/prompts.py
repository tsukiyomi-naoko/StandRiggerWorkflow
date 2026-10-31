"""Edit-prompt library for Qwen-Image-Edit-2511 (and other instruction editors).

Every preset returns panels in the order StandRig's modeling contract
(AGENTS.md / examples/reference-manifest.template.json) expects. Each panel
carries the Live2D-style parameter extreme it represents, so the reference
manifest can pin "intended parameter extremes" per panel.

Direction words are always given from the *viewer's* point of view, because
"left/right" from a parameter name is ambiguous (StandRig's guide: "Do not infer
camera direction/sign from a parameter name").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

KEEP = (
    "Keep exactly the same character: same face, eye shape and eye color, hairstyle, hair color, "
    "outfit, accessories, line-art thickness, cel-shading style, color palette and lighting. "
    "Keep the same canvas size, framing and scale, and a plain flat white background. "
    "Do not add new objects, text, effects or shadows on the background."
)


@dataclass
class Panel:
    id: str
    prompt: Optional[str]  # None = produced deterministically (neutral copy / rotation)
    extremes: Dict[str, float] = field(default_factory=dict)
    direction: str = ""
    mask_prompt: str = ""  # SAM3 text for the region that changes (used for PSD variant layers)
    method: str = "ai"  # ai | source | rotate


def _p(text: str, subject: str, extra: str) -> str:
    s = text.replace("{subject}", subject or "the character")
    s = f"{s} {KEEP}"
    if extra.strip():
        s = f"{s} {extra.strip()}"
    return s


def face_targets(subject: str, yaw: float, pitch: float, roll: float, roll_mode: str, extra: str) -> List[Panel]:
    y, p, r = int(round(yaw)), int(round(pitch)), int(round(roll))
    panels = [
        Panel("neutral", None, {"ParamAngleX": 0, "ParamAngleY": 0, "ParamAngleZ": 0},
              "approved source artwork, front facing", method="source"),
        Panel("yaw-min", _p(
            f"Turn {{subject}}'s head {y} degrees so the face looks toward the viewer's left. Only the head "
            "rotates; the neck base and shoulders stay where they are. Use correct 3D perspective: the face "
            "on the viewer's-left side narrows and becomes the far side, the nose points to the viewer's left, "
            "the far eye becomes narrower, and the bangs and side hair rotate together with the skull.",
            subject, extra), {"ParamAngleX": -y}, f"head yaw {y} deg toward viewer's left"),
        Panel("yaw-max", _p(
            f"Turn {{subject}}'s head {y} degrees so the face looks toward the viewer's right. Only the head "
            "rotates; the neck base and shoulders stay where they are. Use correct 3D perspective: the face "
            "on the viewer's-right side narrows and becomes the far side, the nose points to the viewer's right, "
            "the far eye becomes narrower, and the bangs and side hair rotate together with the skull.",
            subject, extra), {"ParamAngleX": y}, f"head yaw {y} deg toward viewer's right"),
        Panel("pitch-min", _p(
            f"Tilt {{subject}}'s head down by {p} degrees so the face looks downward. Only the head tilts; "
            "the neck base and shoulders stay in place. Perspective: more of the top of the head and the hair "
            "crown is visible, the eyes and mouth move lower on the face, the chin tucks toward the neck and "
            "the eyelids cover more of the eyes.", subject, extra),
              {"ParamAngleY": -p}, f"head pitch {p} deg down"),
        Panel("pitch-max", _p(
            f"Tilt {{subject}}'s head up by {p} degrees so the face looks upward. Only the head tilts; the neck "
            "base and shoulders stay in place. Perspective: the underside of the jaw and chin becomes visible, "
            "the eyes and mouth move higher on the face, the neck stretches and the bangs lift slightly away "
            "from the forehead.", subject, extra),
              {"ParamAngleY": p}, f"head pitch {p} deg up"),
    ]
    for pid, sign, word in (("roll-min", -1, "left"), ("roll-max", 1, "right")):
        if roll_mode == "ai":
            panels.append(Panel(pid, _p(
                f"Tilt {{subject}}'s head sideways by {r} degrees so the top of the head leans toward the "
                f"viewer's {word}, pivoting at the neck. The face stays front facing (no turning), the shoulders "
                "stay level and the hair hangs naturally with gravity.", subject, extra),
                {"ParamAngleZ": sign * r}, f"head roll {r} deg, crown toward viewer's {word}"))
        else:
            panels.append(Panel(pid, None, {"ParamAngleZ": sign * r},
                                f"head roll {r} deg, crown toward viewer's {word} (exact rotation of neutral about the chin/neck pivot)",
                                method="rotate"))
    return panels


def body_targets(subject: str, yaw: float, pitch: float, roll: float, roll_mode: str, extra: str) -> List[Panel]:
    y, p, r = int(round(yaw)), int(round(pitch)), int(round(roll))
    panels = [
        Panel("neutral", None, {"ParamBodyAngleX": 0, "ParamBodyAngleY": 0, "ParamBodyAngleZ": 0},
              "approved source artwork, standing", method="source"),
        Panel("body-yaw-min", _p(
            f"Rotate {{subject}}'s upper body {y} degrees from the waist so the chest turns toward the viewer's "
            "left. The hips, legs and feet stay planted; the head turns along with the chest by a smaller amount; "
            "the far shoulder becomes narrower and the near shoulder wider with correct perspective, and the "
            "arms and sleeves follow the shoulders.", subject, extra),
              {"ParamBodyAngleX": -y}, f"body yaw {y} deg toward viewer's left"),
        Panel("body-yaw-max", _p(
            f"Rotate {{subject}}'s upper body {y} degrees from the waist so the chest turns toward the viewer's "
            "right. The hips, legs and feet stay planted; the head turns along with the chest by a smaller amount; "
            "the far shoulder becomes narrower and the near shoulder wider with correct perspective, and the "
            "arms and sleeves follow the shoulders.", subject, extra),
              {"ParamBodyAngleX": y}, f"body yaw {y} deg toward viewer's right"),
        Panel("body-pitch-min", _p(
            f"Make {{subject}} lean the upper body forward by {p} degrees from the waist, a slight bow toward the "
            "viewer. The legs and feet stay planted, the shoulders come forward, the chest is foreshortened and the "
            "hair falls forward slightly.", subject, extra),
              {"ParamBodyAngleY": -p}, f"body pitch {p} deg forward"),
        Panel("body-pitch-max", _p(
            f"Make {{subject}} lean the upper body backward by {p} degrees from the waist, chest lifted. The legs "
            "and feet stay planted, the shoulders go back, the neck and the underside of the chin become slightly "
            "more visible.", subject, extra),
              {"ParamBodyAngleY": p}, f"body pitch {p} deg back"),
    ]
    for pid, sign, word in (("body-roll-min", -1, "left"), ("body-roll-max", 1, "right")):
        if roll_mode == "rotate":
            panels.append(Panel(pid, None, {"ParamBodyAngleZ": sign * r},
                                f"body roll {r} deg toward viewer's {word} (exact rotation about the hip pivot)",
                                method="rotate"))
        else:
            panels.append(Panel(pid, _p(
                f"Make {{subject}} bend the upper body sideways by {r} degrees so the shoulders tilt and the head "
                f"leans toward the viewer's {word}, pivoting at the waist. The hips, legs and feet stay planted and "
                "level; the skirt/clothes and hair hang naturally with gravity.", subject, extra),
                {"ParamBodyAngleZ": sign * r}, f"body roll {r} deg toward viewer's {word}"))
    return panels


CONNECTIONS = [
    ("hair-roots-back-hair",
     "Remove the front bangs and the side locks of {subject} so the forehead, the hairline and the hair roots "
     "are fully visible, and draw the back hair as one complete mass behind the head and neck, including its "
     "darker inner side. The face, eyes and back hair otherwise stay identical.",
     "hair", (0.00, 0.00, 1.00, 0.42)),
    ("jaw-neck",
     "Raise {subject}'s chin slightly and draw the complete neck continuing up under the jaw line, with the "
     "neck skin extending behind the jaw so there is no gap between the jaw and the neck.",
     "neck", (0.10, 0.08, 0.90, 0.42)),
    ("neck-collar-shoulders",
     "Show the full neck, collarbones and shoulder line of {subject} where they meet the collar: draw the skin "
     "and the inside of the collar fully, as if the collar were slightly lifted, keeping the same outfit.",
     "collar", (0.00, 0.15, 1.00, 0.50)),
    ("sleeves-arms",
     "Raise {subject}'s arms slightly away from the body and draw the complete shoulder joints and the arms "
     "continuing inside the sleeve openings, including the inner side of the sleeves.",
     "arm", (0.00, 0.18, 1.00, 0.70)),
    ("waist-skirt-thighs",
     "Draw the inner lining of {subject}'s skirt or lower garment and the waist/hip area underneath it, with "
     "modest shorts or leggings, so the legs continue cleanly under the hem with no gap. Keep it non-revealing.",
     "skirt", (0.00, 0.40, 1.00, 0.80)),
    ("thighs-lower-legs",
     "Bend {subject}'s knees very slightly and draw the complete knee joints so the thighs connect smoothly to "
     "the lower legs, including any socks or leggings overlapping at the knee.",
     "knee", (0.00, 0.55, 1.00, 1.00)),
]


def connection_targets(subject: str, extra: str) -> List[Panel]:
    out = []
    for pid, text, mask, band in CONNECTIONS:
        pan = Panel(pid, _p(text, subject, extra), {}, f"hidden overlap art for {pid}", mask_prompt=mask)
        pan.extremes = {"cropBand": list(band)}
        out.append(pan)
    return out


EXPRESSIONS = [
    # id, prompt, mask region, parameter hint
    ("eyes-closed", "Make {subject} close both eyes gently in a relaxed blink: the upper eyelids come down to "
     "a soft closed curve with eyelashes, no other change to the face.", "eyes", {"ParamEyeLOpen": 0, "ParamEyeROpen": 0}),
    ("eyes-half", "Make {subject}'s eyes half closed (sleepy), upper eyelids covering the top half of the irises.",
     "eyes", {"ParamEyeLOpen": 0.5, "ParamEyeROpen": 0.5}),
    ("eyes-smile", "Make {subject} smile with closed, upward-curved happy eyes (^ ^ shape), mouth closed in a gentle smile.",
     "eyes", {"ParamEyeLSmile": 1, "ParamEyeRSmile": 1}),
    ("mouth-a", "Make {subject} say the vowel 'A' (as in 'ah'): mouth wide open vertically, showing the upper teeth, "
     "tongue and dark mouth interior. Nothing else changes.", "mouth", {"ParamMouthOpenY": 1, "ParamMouthForm": 0}),
    ("mouth-i", "Make {subject} say the vowel 'I' (as in 'ee'): lips stretched wide sideways, mouth slightly open, "
     "teeth visible. Nothing else changes.", "mouth", {"ParamMouthOpenY": 0.3, "ParamMouthForm": 1}),
    ("mouth-u", "Make {subject} say the vowel 'U' (as in 'oo'): lips pushed forward into a small rounded opening. "
     "Nothing else changes.", "mouth", {"ParamMouthOpenY": 0.35, "ParamMouthForm": -1}),
    ("mouth-e", "Make {subject} say the vowel 'E' (as in 'eh'): mouth half open and slightly wide, upper teeth "
     "visible. Nothing else changes.", "mouth", {"ParamMouthOpenY": 0.6, "ParamMouthForm": 0.5}),
    ("mouth-o", "Make {subject} say the vowel 'O' (as in 'oh'): mouth open in a tall rounded oval showing the dark "
     "interior. Nothing else changes.", "mouth", {"ParamMouthOpenY": 0.8, "ParamMouthForm": -0.6}),
    ("mouth-smile", "Make {subject} smile with the mouth open happily, corners raised, eyes unchanged.",
     "mouth", {"ParamMouthOpenY": 0.5, "ParamMouthForm": 1}),
    ("brows-angry", "Make {subject} frown angrily: eyebrows pulled down and together at the center. Eyes and mouth unchanged.",
     "eyebrows", {"ParamBrowLAngle": -1, "ParamBrowRAngle": -1}),
    ("brows-sad", "Make {subject} look worried and sad: inner ends of the eyebrows raised. Eyes and mouth unchanged.",
     "eyebrows", {"ParamBrowLAngle": 1, "ParamBrowRAngle": 1}),
    ("blush", "Add a soft anime blush (light pink cheeks with fine diagonal hatching) to {subject}'s face. "
     "Nothing else changes.", "cheeks", {"ParamCheek": 1}),
]


def expressions(subject: str, extra: str, selection: str) -> List[Panel]:
    wanted = [s.strip() for s in (selection or "").replace("\n", ",").split(",") if s.strip()]
    out = [Panel("neutral", None, {}, "approved source artwork", method="source")]
    for pid, text, mask, params in EXPRESSIONS:
        if wanted and pid not in wanted:
            continue
        out.append(Panel(pid, _p(text, subject, extra), params, pid, mask_prompt=mask))
    return out


# 8 azimuths the fal Multiple-Angles LoRA was trained on. Prompt format from the
# model card: "<sks> {azimuth} {elevation} {distance}".
LORA_AZIMUTHS = [
    ("front", "front view"), ("front-right", "front-right quarter view"), ("right", "right side view"),
    ("back-right", "back-right quarter view"), ("back", "back view"), ("back-left", "back-left quarter view"),
    ("left", "left side view"), ("front-left", "front-left quarter view"),
]


def turnaround(subject: str, extra: str, mode: str, shot: str) -> List[Panel]:
    out = []
    if mode == "multiangle_lora":
        for pid, az in LORA_AZIMUTHS:
            if pid == "front":
                out.append(Panel("front", None, {}, "approved source artwork", method="source"))
                continue
            # The LoRA expects its fixed trigger format; keep the identity hint short.
            out.append(Panel(pid, f"<sks> {az} eye-level shot {shot}", {}, az))
    else:
        views = [
            ("front", None),
            ("front-left", "Rotate the camera 45 degrees to the left around {subject}."),
            ("left", "Rotate the camera 90 degrees to the left around {subject}, showing a full side profile."),
            ("back", "Rotate the camera 180 degrees around {subject}, showing the back view."),
            ("right", "Rotate the camera 90 degrees to the right around {subject}, showing a full side profile."),
            ("front-right", "Rotate the camera 45 degrees to the right around {subject}."),
        ]
        for pid, text in views:
            if text is None:
                out.append(Panel(pid, None, {}, "approved source artwork", method="source"))
            else:
                out.append(Panel(pid, _p(text, subject, extra), {}, pid))
    return out


CONTRACT_ORDER = {
    "face": ["neutral", "yaw-min", "yaw-max", "pitch-min", "pitch-max", "roll-min", "roll-max"],
    "body": ["neutral", "body-yaw-min", "body-yaw-max", "body-pitch-min", "body-pitch-max", "body-roll-min", "body-roll-max"],
    "connections": [c[0] for c in CONNECTIONS],
}

NORMALIZE_PROMPT = (
    "Redraw {subject} as a clean character model reference for Live2D rigging: full body, standing straight and "
    "facing the viewer directly, head level and looking at the viewer, eyes fully open, mouth closed in a neutral "
    "expression, arms relaxed and held slightly away from the torso so the sides of the body are visible, legs "
    "slightly apart, no props in the hands, nothing overlapping the face. " + KEEP
)
