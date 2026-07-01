"""
ComfyUI-SpeechBubble – nodes.py
================================
Adds customizable speech bubbles to images inside a ComfyUI workflow.

Features
--------
* Three bubble shapes  : ellipse / rounded_rectangle / rectangle
* Tail (꼬리)          : position controlled via (tail_x, tail_y) coordinates
* HandDrawn mode       : perturbs the outline so it looks hand-sketched
* Dual output          : composited RGB image  +  bubble RGBA layer  +  bubble MASK
* JSON settings        : serialize / deserialize bubble layout as a JSON string

Font setup
----------
Place any .ttf / .otf / .ttc font files inside
    <this_node_dir>/assets/font/
and they will appear in the font_name dropdown automatically.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ─── Paths ────────────────────────────────────────────────────────────────────
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_FONT_DIR = os.path.join(_NODE_DIR, "assets", "font")

# ─── Tensor ↔ PIL helpers ─────────────────────────────────────────────────────

def _t2p(t: torch.Tensor) -> Image.Image:
    """Tensor BHWC [0,1] (single batch slice) → PIL RGB."""
    arr = t.squeeze(0).detach().cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA").convert("RGB")
    return Image.fromarray(arr, "RGB")


def _t2p_rgba(t: torch.Tensor) -> Image.Image:
    """Tensor BHWC [0,1] (single batch slice) → PIL RGBA."""
    arr = t.squeeze(0).detach().cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGBA")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr, "RGB").convert("RGBA")



def _p2t_rgb(img: Image.Image) -> torch.Tensor:
    """PIL → BHWC float32 tensor (3-channel RGB)."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _p2t_rgba(img: Image.Image) -> torch.Tensor:
    """PIL → BHWC float32 tensor (4-channel RGBA)."""
    arr = np.array(img.convert("RGBA")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ─── Font utilities ───────────────────────────────────────────────────────────

def _list_fonts() -> List[str]:
    fonts = ["default"]
    if os.path.isdir(_FONT_DIR):
        for f in sorted(os.listdir(_FONT_DIR)):
            if f.lower().endswith((".ttf", ".otf", ".ttc")):
                fonts.append(f)
    return fonts


def _load_font(name: str, size: int) -> ImageFont.ImageFont:
    if name and name != "default":
        p = os.path.join(_FONT_DIR, name)
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    if os.path.isdir(_FONT_DIR):
        for f in os.listdir(_FONT_DIR):
            if f.lower().endswith((".ttf", ".otf", ".ttc")):
                try:
                    return ImageFont.truetype(os.path.join(_FONT_DIR, f), size)
                except Exception:
                    pass
    return ImageFont.load_default()


# ─── Color helper ─────────────────────────────────────────────────────────────

def _hex_rgba(hex_str: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    h = hex_str.strip().lstrip("#")
    try:
        if len(h) == 6:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)
        if len(h) == 8:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16))
    except ValueError:
        pass
    return (0, 0, 0, alpha)


# ─── Geometry ─────────────────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def _gen_ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 80) -> List[Tuple[float, float]]:
    """CCW ellipse points."""
    return [
        (cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def _gen_rounded_rect(x1: float, y1: float, x2: float, y2: float, r: float, k: int = 15) -> List[Tuple[float, float]]:
    """CCW rounded-rectangle points."""
    r = max(1.0, min(r, (x2 - x1) / 2 * 0.9, (y2 - y1) / 2 * 0.9))
    pts: List[Tuple[float, float]] = []
    for ox, oy, a0, a1 in [
        (x1 + r, y1 + r, math.pi,        3 * math.pi / 2),
        (x2 - r, y1 + r, 3 * math.pi / 2, 2 * math.pi),
        (x2 - r, y2 - r, 0.0,             math.pi / 2),
        (x1 + r, y2 - r, math.pi / 2,     math.pi),
    ]:
        for i in range(k):
            a = a0 + (a1 - a0) * i / k
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    return pts


def _gen_rect(x1: float, y1: float, x2: float, y2: float, n: int = 20) -> List[Tuple[float, float]]:
    """CCW rectangle points (uniformly sampled on each side)."""
    pts: List[Tuple[float, float]] = []
    for i in range(n): pts.append((x1 + (x2 - x1) * i / n, y1))
    for i in range(n): pts.append((x2, y1 + (y2 - y1) * i / n))
    for i in range(n): pts.append((x2 - (x2 - x1) * i / n, y2))
    for i in range(n): pts.append((x1, y2 - (y2 - y1) * i / n))
    return pts


def _build_polygon(
    shape: str,
    x1: float, y1: float, x2: float, y2: float,
    tx: float, ty: float,
    tail_half: int,
) -> List[Tuple[float, float]]:
    """
    Return a closed polygon that represents the bubble outline + tail tip.

    Algorithm
    ---------
    1. Generate N CCW points for the bubble body.
    2. Find the point (``attach``) whose angle from bubble centre is
       closest to the tail direction.
    3. Mark a "skip zone" of width 2*tail_half around ``attach``.
    4. Traverse CCW skipping that zone, then insert the tail tip.
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    if shape == "ellipse":
        base = _gen_ellipse(cx, cy, (x2 - x1) / 2, (y2 - y1) / 2, 80)
    elif shape == "rounded_rectangle":
        r = min((x2 - x1), (y2 - y1)) * 0.18
        base = _gen_rounded_rect(x1, y1, x2, y2, r, 15)
    else:
        base = _gen_rect(x1, y1, x2, y2, 20)

    N = len(base)

    # Tail direction angle
    tail_angle = math.atan2(ty - cy, tx - cx)

    # Index of the bubble point closest in direction to the tail
    attach = min(
        range(N),
        key=lambda i: _angle_diff(math.atan2(base[i][1] - cy, base[i][0] - cx), tail_angle),
    )

    h = max(1, min(tail_half, N // 4))

    # Skip zone boundaries (skip_start..skip_end in CCW index order)
    skip_start = (attach - h) % N   # first skipped index
    skip_end   = (attach + h) % N   # last  skipped index

    # Build polygon: start just AFTER skip_end, traverse CCW, stop just BEFORE skip_start
    start_idx = (skip_end + 1) % N
    end_idx   = (skip_start - 1 + N) % N

    poly: List[Tuple[float, float]] = []
    i = start_idx
    for _ in range(N + 1):
        poly.append(base[i])
        if i == end_idx:
            break
        i = (i + 1) % N

    poly.append((float(tx), float(ty)))  # tail tip
    return poly


# ─── HandDrawn perturbation ───────────────────────────────────────────────────

def _perturb(pts: List[Tuple[float, float]], sigma: float, rng: random.Random) -> List[Tuple[float, float]]:
    return [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma)) for x, y in pts]


# ─── Text wrapping ────────────────────────────────────────────────────────────

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> List[str]:
    if max_w <= 0:
        max_w = 1
    lines: List[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        
        cur_line = ""
        words = para.split(" ")
        for word in words:
            if not word:
                if cur_line:
                    test_space = cur_line + " "
                    bb = draw.textbbox((0, 0), test_space, font=font)
                    if bb[2] - bb[0] <= max_w:
                        cur_line = test_space
                    else:
                        lines.append(cur_line)
                        cur_line = ""
                continue

            test_line = (cur_line + " " + word) if cur_line else word
            bb = draw.textbbox((0, 0), test_line, font=font)
            if bb[2] - bb[0] <= max_w:
                cur_line = test_line
            else:
                bb_word = draw.textbbox((0, 0), word, font=font)
                if bb_word[2] - bb_word[0] <= max_w:
                    if cur_line:
                        lines.append(cur_line)
                    cur_line = word
                else:
                    if cur_line:
                        test_space = cur_line + " "
                        bb_space = draw.textbbox((0, 0), test_space, font=font)
                        if bb_space[2] - bb_space[0] <= max_w:
                            cur_line = test_space
                        else:
                            lines.append(cur_line)
                            cur_line = ""
                    
                    for char in word:
                        test_char = cur_line + char
                        bb_char = draw.textbbox((0, 0), test_char, font=font)
                        if bb_char[2] - bb_char[0] <= max_w:
                            cur_line = test_char
                        else:
                            if cur_line:
                                lines.append(cur_line)
                            cur_line = char
        if cur_line:
            lines.append(cur_line)
    return lines


# ─── Core render ──────────────────────────────────────────────────────────────

def render_bubble(
    base: Image.Image,
    text: str,
    bx: int, by: int, bw: int, bh: int,
    tx: int, ty: int,
    shape: str,
    fill_hex: str, border_hex: str, text_hex: str,
    border_w: int,
    font_name: str, font_size: int,
    padding: int, line_sp: int,
    tail_ratio: float,
    handdrawn: bool, hd_str: float,
    opacity: int,
    seed: int,
    bubble_pil: Image.Image = None,
) -> Tuple[Image.Image, Image.Image]:
    """
    Render a speech bubble onto *base*.

    Returns
    -------
    composited_rgb : PIL RGB image  (base + bubble)
    bubble_rgba    : PIL RGBA image (bubble layer only, transparent background)
    """
    W, H = base.size

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer, "RGBA")

    fill_c   = _hex_rgba(fill_hex,   opacity)
    border_c = _hex_rgba(border_hex, 255)
    text_c   = _hex_rgba(text_hex,   255)

    x2, y2 = bx + bw, by + bh
    cx, cy  = bx + bw / 2.0, by + bh / 2.0

    if bubble_pil is not None:
        resized_bubble = bubble_pil.resize((bw, bh), Image.LANCZOS)
        if opacity < 255:
            r, g, b, a = resized_bubble.split()
            a = a.point(lambda p: int(p * (opacity / 255.0)))
            resized_bubble = Image.merge("RGBA", (r, g, b, a))
        layer.paste(resized_bubble, (bx, by), mask=resized_bubble)
    else:
        # Approximate tail attachment width in polygon points
        tail_half = max(1, int(80 * tail_ratio * 0.5))

        poly = _build_polygon(shape, bx, by, x2, y2, tx, ty, tail_half)
        rng  = random.Random(seed if seed >= 0 else None)

        if handdrawn and hd_str > 0:
            # Fill once with the base (non-perturbed) polygon
            draw.polygon(poly, fill=fill_c)
            # Sketch lines: 3 passes with increasing perturbation
            if border_w > 0:
                for k in range(3):
                    p = _perturb(poly, hd_str * (0.5 + 0.4 * k), rng)
                    draw.line(p + [p[0]], fill=border_c, width=border_w)
        else:
            draw.polygon(poly, fill=fill_c)
            if border_w > 0:
                draw.line(poly + [poly[0]], fill=border_c, width=border_w)

    # ── Text ─────────────────────────────────────────────────────────────────
    font   = _load_font(font_name, font_size)
    lines  = _wrap_text(draw, text, font, bw - 2 * padding)

    bb0    = draw.textbbox((0, 0), "Ag", font=font)
    line_h = (bb0[3] - bb0[1]) + line_sp
    total_h = len(lines) * line_h - line_sp

    sy = cy - total_h / 2
    for idx, line in enumerate(lines):
        if not line:
            continue
        bb  = draw.textbbox((0, 0), line, font=font)
        lx  = cx - (bb[2] - bb[0]) / 2
        draw.text((lx, sy + idx * line_h), line, fill=text_c, font=font)

    # ── Composite ────────────────────────────────────────────────────────────
    comp_rgb = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    return comp_rgb, layer


# ─── ComfyUI Node: SpeechBubble ───────────────────────────────────────────────

class SpeechBubbleNode:
    """
    🗨️ Speech Bubble
    ─────────────────
    Adds a speech bubble with a movable tail to an image.

    Outputs
    -------
    composited  : Input image with the bubble rendered on top (RGB).
    bubble_rgba : The bubble layer only on a transparent background (RGBA).
    bubble_mask : Alpha mask of the bubble (values 0–1).
    """

    RETURN_TYPES  = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES  = ("composited", "bubble_rgba", "bubble_mask")
    FUNCTION      = "run"
    CATEGORY      = "image/speech_bubble"
    OUTPUT_NODE   = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":       ("IMAGE",),
                "text":        ("STRING", {"default": "Hello!", "multiline": True}),
                # ── Bubble geometry ──────────────────────────────────────
                "bubble_x":    ("INT", {"default": 50,  "min": -8192, "max": 8192, "step": 1}),
                "bubble_y":    ("INT", {"default": 50,  "min": -8192, "max": 8192, "step": 1}),
                "bubble_w":    ("INT", {"default": 260, "min": 10,    "max": 8192, "step": 1}),
                "bubble_h":    ("INT", {"default": 130, "min": 10,    "max": 8192, "step": 1}),
                # ── Tail tip position (꼬리 끝 좌표) ─────────────────────
                "tail_x":      ("INT", {"default": 180, "min": -8192, "max": 8192, "step": 1}),
                "tail_y":      ("INT", {"default": 240, "min": -8192, "max": 8192, "step": 1}),
                # ── Appearance ───────────────────────────────────────────
                "shape":       (["ellipse", "rounded_rectangle", "rectangle"],),
                "fill_color":  ("STRING", {"default": "#FFFFFF"}),
                "border_color":("STRING", {"default": "#000000"}),
                "text_color":  ("STRING", {"default": "#000000"}),
                "border_width":("INT", {"default": 3, "min": 0, "max": 30}),
                "opacity":     ("INT", {"default": 255, "min": 0, "max": 255}),
                # ── Font & text layout ───────────────────────────────────
                "font_name":   (_list_fonts(),),
                "font_size":   ("INT", {"default": 22, "min": 6,  "max": 200}),
                "padding":     ("INT", {"default": 15, "min": 0,  "max": 200}),
                "line_spacing":("INT", {"default": 4,  "min": 0,  "max": 50}),
                # ── Tail width ───────────────────────────────────────────
                "tail_width":  ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
                # ── HandDrawn effect ─────────────────────────────────────
                "handdrawn":   ("BOOLEAN", {"default": False}),
                "handdrawn_strength": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "seed":        ("INT", {"default": 42, "min": -1, "max": 999_999_999}),
            },
            "optional": {
                # Connect custom bubble image here
                "bubble_image": ("IMAGE",),
                # Connect a SpeechBubbleSettings node here to override geometry from JSON
                "settings_json": ("STRING", {"forceInput": True}),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        text: str,
        bubble_x: int, bubble_y: int, bubble_w: int, bubble_h: int,
        tail_x: int, tail_y: int,
        shape: str,
        fill_color: str, border_color: str, text_color: str,
        border_width: int, opacity: int,
        font_name: str, font_size: int,
        padding: int, line_spacing: int,
        tail_width: float,
        handdrawn: bool, handdrawn_strength: float,
        seed: int,
        bubble_image: torch.Tensor = None,
        settings_json: str = "",
    ):
        # Override geometry from JSON if provided
        if settings_json:
            try:
                cfg = json.loads(settings_json)
                bubble_x = cfg.get("bubble_x", bubble_x)
                bubble_y = cfg.get("bubble_y", bubble_y)
                bubble_w = cfg.get("bubble_w", bubble_w)
                bubble_h = cfg.get("bubble_h", bubble_h)
                tail_x   = cfg.get("tail_x",   tail_x)
                tail_y   = cfg.get("tail_y",    tail_y)
                text     = cfg.get("text",      text)
            except Exception as e:
                print(f"[SpeechBubble] Warning: could not parse settings_json – {e}")

        comp_batch: List[torch.Tensor]   = []
        rgba_batch: List[torch.Tensor]   = []
        mask_batch: List[torch.Tensor]   = []

        for b in range(image.shape[0]):
            pil = _t2p(image[b : b + 1])

            bubble_pil = None
            if bubble_image is not None:
                bubble_idx = b if b < bubble_image.shape[0] else 0
                bubble_pil = _t2p_rgba(bubble_image[bubble_idx : bubble_idx + 1])

            comp, blay = render_bubble(
                pil, text,
                bubble_x, bubble_y, bubble_w, bubble_h,
                tail_x, tail_y, shape,
                fill_color, border_color, text_color,
                border_width, font_name, font_size,
                padding, line_spacing, tail_width,
                handdrawn, handdrawn_strength,
                opacity, seed + b,
                bubble_pil,
            )

            comp_batch.append(_p2t_rgb(comp))
            rgba_batch.append(_p2t_rgba(blay))

            # Alpha mask [1, H, W] → will be concatenated to [B, H, W]
            alpha_arr = np.array(blay)[:, :, 3].astype(np.float32) / 255.0
            mask_batch.append(torch.from_numpy(alpha_arr).unsqueeze(0))

        return (
            torch.cat(comp_batch, dim=0),   # [B, H, W, 3]
            torch.cat(rgba_batch, dim=0),   # [B, H, W, 4]
            torch.cat(mask_batch, dim=0),   # [B, H, W]
        )


# ─── ComfyUI Node: SpeechBubbleSettings ──────────────────────────────────────

class SpeechBubbleSettingsNode:
    """
    🗨️ Speech Bubble Settings
    ──────────────────────────
    Serialise bubble position / size / tail / text as a JSON string that
    can be fed into the *settings_json* input of a SpeechBubble node.
    Useful for storing and reusing a layout without touching the main node.
    """

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("settings_json",)
    FUNCTION      = "run"
    CATEGORY      = "image/speech_bubble"
    OUTPUT_NODE   = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bubble_x": ("INT", {"default": 50,  "min": -8192, "max": 8192}),
                "bubble_y": ("INT", {"default": 50,  "min": -8192, "max": 8192}),
                "bubble_w": ("INT", {"default": 260, "min": 10,    "max": 8192}),
                "bubble_h": ("INT", {"default": 130, "min": 10,    "max": 8192}),
                "tail_x":   ("INT", {"default": 180, "min": -8192, "max": 8192}),
                "tail_y":   ("INT", {"default": 240, "min": -8192, "max": 8192}),
                "text":     ("STRING", {"default": "Hello!", "multiline": True}),
            }
        }

    def run(
        self,
        bubble_x: int, bubble_y: int,
        bubble_w: int, bubble_h: int,
        tail_x: int,   tail_y: int,
        text: str,
    ):
        payload = dict(
            bubble_x=bubble_x, bubble_y=bubble_y,
            bubble_w=bubble_w, bubble_h=bubble_h,
            tail_x=tail_x,     tail_y=tail_y,
            text=text,
        )
        return (json.dumps(payload, ensure_ascii=False),)


# ─── ComfyUI Node: SpeechBubbleExtractor ──────────────────────────────────────

class SpeechBubbleExtractorNode:
    """
    🗨️ Speech Bubble Extractor
    ──────────────────────────
    Takes a masked speech bubble image (with text), erases the text to create
    an empty speech bubble, optionally renders new text, and composites it
    onto the original image.
    """

    RETURN_TYPES  = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("composited", "empty_bubble", "masked_bubble")
    FUNCTION      = "run"
    CATEGORY      = "image/speech_bubble"
    OUTPUT_NODE   = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":               ("IMAGE",),
                "masked_bubble_image": ("IMAGE",),
                "font_name":           (_list_fonts(),),
                "font_size":           ("INT", {"default": 22, "min": 6,  "max": 200}),
                "font_color":          ("STRING", {"default": "#000000"}),
                "padding":             ("INT", {"default": 15, "min": 0,  "max": 200}),
                "line_spacing":        ("INT", {"default": 4,  "min": 0,  "max": 50}),
                "mask_blur":           ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "text":                ("STRING", {"default": "", "multiline": True}),
                "erosion_margin":      ("INT", {"default": 15, "min": 0, "max": 100}),
                "text_threshold":      ("INT", {"default": 150, "min": 0, "max": 255}),
            }
        }

    def run(
        self,
        image: torch.Tensor,
        masked_bubble_image: torch.Tensor,
        font_name: str,
        font_size: int,
        font_color: str,
        padding: int,
        line_spacing: int,
        mask_blur: float,
        text: str = "",
        erosion_margin: int = 15,
        text_threshold: int = 150,
    ):
        import cv2
        from PIL import ImageFilter

        comp_batch: List[torch.Tensor]   = []
        empty_batch: List[torch.Tensor]  = []
        masked_batch: List[torch.Tensor] = []

        for b in range(image.shape[0]):
            # 1. Convert tensors to PIL Images
            orig_pil = _t2p(image[b : b + 1])
            
            bubble_idx = b if b < masked_bubble_image.shape[0] else 0
            bubble_pil = _t2p_rgba(masked_bubble_image[bubble_idx : bubble_idx + 1])

            # Save input bubble for output
            masked_batch.append(_p2t_rgba(bubble_pil))

            # 2. Extract bubble mask
            bubble_rgba = bubble_pil.convert("RGBA")
            r, g, b_ch, a = bubble_rgba.split()
            a_np = np.array(a)

            # If alpha is uniform, threshold based on color to find bubble
            if a_np.max() == a_np.min():
                rgb_np = np.array(bubble_rgba.convert("RGB"))
                mask_np = np.any(rgb_np < 250, axis=2).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_filled = np.zeros_like(mask_np)
                if contours:
                    cv2.drawContours(mask_filled, contours, -1, 255, -1)
                mask_np = mask_filled
            else:
                mask_np = a_np

            # 3. Erase text via erosion + threshold + inpainting
            if erosion_margin > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_margin, erosion_margin))
                eroded_mask = cv2.erode(mask_np, kernel, iterations=1)
            else:
                eroded_mask = mask_np

            gray = np.array(bubble_rgba.convert("L"))
            text_mask = (eroded_mask > 0) & (gray < text_threshold)

            # Inpaint text area
            img_bgr = cv2.cvtColor(np.array(bubble_rgba.convert("RGB")), cv2.COLOR_RGB2BGR)
            text_mask_uint8 = text_mask.astype(np.uint8) * 255
            
            if np.any(text_mask_uint8 > 0):
                inpainted_bgr = cv2.inpaint(img_bgr, text_mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
            else:
                inpainted_rgb = np.array(bubble_rgba.convert("RGB"))

            # Reconstruct the empty bubble RGBA image
            empty_np = np.array(bubble_rgba)
            empty_np[:, :, 0:3] = inpainted_rgb
            # Ensure the bubble interior mask region has alpha = 255
            empty_np[eroded_mask > 0, 3] = 255
            empty_bubble = Image.fromarray(empty_np, "RGBA")

            # 4. Draw new text if provided
            if text.strip():
                draw = ImageDraw.Draw(empty_bubble, "RGBA")
                text_c = _hex_rgba(font_color, 255)

                # ── Helper: find safe per-row wrap width ──────────────────────
                def _row_safe_wrap_w(mask: np.ndarray, y0: int, y1: int, pad: int) -> int:
                    """Return the minimum contiguous run width in [y0,y1] rows, minus 2*pad."""
                    y0c = max(0, y0)
                    y1c = min(mask.shape[0], y1)
                    if y0c >= y1c:
                        return 1
                    region = mask[y0c:y1c, :]
                    col_on = np.any(region > 0, axis=0)
                    cols = np.where(col_on)[0]
                    if len(cols) == 0:
                        return 1
                    return max(1, int(cols[-1] - cols[0]) - 2 * pad)

                # ── Helper: draw text block in one region ─────────────────────
                def _draw_text_in_region(
                    drw: ImageDraw.ImageDraw,
                    region_mask: np.ndarray,
                    full_mask: np.ndarray,
                    region_text: str,
                    f_name: str, f_size: int, f_color,
                    pad: int, sp: int,
                    test_only: bool = False,
                ) -> Tuple[bool, int]:
                    """
                    Draw (or test-draw) region_text centred inside the region defined
                    by region_mask, auto-shrinking font until no overflow.
                    Returns (overflow_occurred, final_font_size).
                    """
                    row_idx = np.where(np.sum(region_mask > 0, axis=1) > 0)[0]
                    if len(row_idx) == 0:
                        return False, f_size

                    ry_min, ry_max = int(row_idx[0]), int(row_idx[-1])

                    # bounding-box fallback cx
                    col_idx_all = np.where(np.any(region_mask > 0, axis=0))[0]
                    fb_cx = (int(col_idx_all[0]) + int(col_idx_all[-1])) / 2.0 if len(col_idx_all) else region_mask.shape[1] / 2.0

                    cur_sz = f_size
                    while cur_sz >= 6:
                        fnt = _load_font(f_name, cur_sz)
                        # wrap based on min row width in the vertical centre slice
                        mid_y0 = ry_min + (ry_max - ry_min) // 4
                        mid_y1 = ry_max - (ry_max - ry_min) // 4
                        ww = _row_safe_wrap_w(region_mask, mid_y0, mid_y1, pad)
                        lns = _wrap_text(drw, region_text, fnt, ww)
                        if not lns:
                            cur_sz -= 1
                            continue

                        bb0 = drw.textbbox((0, 0), "Ag", font=fnt)
                        lh = bb0[3] - bb0[1]
                        th = len(lns) * lh + (len(lns) - 1) * sp

                        rcy = (ry_min + ry_max) / 2.0
                        rsy = rcy - th / 2.0

                        # Test render into a temp L image
                        tmp = Image.new("L", (region_mask.shape[1], region_mask.shape[0]), 0)
                        tmp_drw = ImageDraw.Draw(tmp)
                        for ii, ln in enumerate(lns):
                            if not ln:
                                continue
                            ly = rsy + ii * (lh + sp)
                            # get per-row horizontal center from region mask
                            zym0 = max(0, int(ly))
                            zym1 = min(region_mask.shape[0], int(ly + lh))
                            zm = region_mask[zym0:zym1, :]
                            rc = np.where(np.any(zm > 0, axis=0))[0]
                            if len(rc) > 0:
                                lcx = (int(rc[0]) + pad + int(rc[-1]) - pad) / 2.0
                            else:
                                lcx = fb_cx
                            bb = tmp_drw.textbbox((0, 0), ln, font=fnt)
                            lx = lcx - (bb[2] - bb[0]) / 2.0
                            tmp_drw.text((lx, ly), ln, fill=255, font=fnt)

                        tmp_np = np.array(tmp)
                        # Check overflow against the FULL eroded mask (prevents crossing into neighbour region)
                        overflow = np.any((tmp_np > 0) & (full_mask == 0))
                        if not overflow:
                            # Commit draw if not test_only
                            if not test_only:
                                for ii, ln in enumerate(lns):
                                    if not ln:
                                        continue
                                    ly = rsy + ii * (lh + sp)
                                    zym0 = max(0, int(ly))
                                    zym1 = min(region_mask.shape[0], int(ly + lh))
                                    zm = region_mask[zym0:zym1, :]
                                    rc = np.where(np.any(zm > 0, axis=0))[0]
                                    if len(rc) > 0:
                                        lcx = (int(rc[0]) + pad + int(rc[-1]) - pad) / 2.0
                                    else:
                                        lcx = fb_cx
                                    bb = drw.textbbox((0, 0), ln, font=fnt)
                                    lx = lcx - (bb[2] - bb[0]) / 2.0
                                    drw.text((lx, ly), ln, fill=f_color, font=fnt)
                            return False, cur_sz
                        cur_sz -= 1

                    return True, 6  # could not fit

                # ── Find connected components in eroded mask ──────────────────
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    eroded_mask, connectivity=8
                )

                # Collect non-background components sorted by area (largest first)
                regions = []
                for lbl in range(1, num_labels):
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    if area < 200:
                        continue
                    region_mask_i = (labels == lbl).astype(np.uint8) * 255
                    bx_i = stats[lbl, cv2.CC_STAT_LEFT]
                    by_i = stats[lbl, cv2.CC_STAT_TOP]
                    bw_i = stats[lbl, cv2.CC_STAT_WIDTH]
                    bh_i = stats[lbl, cv2.CC_STAT_HEIGHT]
                    cx_i = bx_i + bw_i / 2.0
                    cy_i = by_i + bh_i / 2.0
                    regions.append({"mask": region_mask_i, "area": area,
                                    "cx": cx_i, "cy": cy_i,
                                    "bx": bx_i, "by": by_i, "bw": bw_i, "bh": bh_i})

                if not regions:
                    # Fallback: treat entire eroded mask as one region
                    regions = [{"mask": eroded_mask, "area": int(np.sum(eroded_mask > 0)),
                                "cx": eroded_mask.shape[1] / 2.0,
                                "cy": eroded_mask.shape[0] / 2.0,
                                "bx": 0, "by": 0,
                                "bw": eroded_mask.shape[1],
                                "bh": eroded_mask.shape[0]}]

                # ── Determine sort axis: horizontal or vertical ───────────────
                if len(regions) > 1:
                    cx_vals = [r["cx"] for r in regions]
                    cy_vals = [r["cy"] for r in regions]
                    spread_x = max(cx_vals) - min(cx_vals)
                    spread_y = max(cy_vals) - min(cy_vals)
                    if spread_x >= spread_y:
                        # side-by-side → sort left-to-right
                        regions.sort(key=lambda r: r["cx"])
                    else:
                        # stacked → sort top-to-bottom
                        regions.sort(key=lambda r: r["cy"])
                else:
                    regions.sort(key=lambda r: r["cy"])

                # ── Split text across regions proportionally by area ──────────
                total_area = sum(r["area"] for r in regions)
                text_parts: List[str] = []
                if "\n\n" in text:
                    raw_parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                    if len(raw_parts) >= len(regions):
                        text_parts = raw_parts
                    else:
                        text_parts = []  # will distribute below
                if not text_parts:
                    words = text.split()
                    total_w = len(words)
                    assigned = 0
                    parts_tmp = []
                    for idx_r, r in enumerate(regions):
                        if idx_r == len(regions) - 1:
                            parts_tmp.append(" ".join(words[assigned:]))
                        else:
                            n = max(1, int(round(r["area"] / total_area * total_w)))
                            parts_tmp.append(" ".join(words[assigned:assigned + n]))
                            assigned += n
                    text_parts = parts_tmp

                # Pad or trim to match region count
                while len(text_parts) < len(regions):
                    text_parts.append("")
                text_parts = text_parts[:len(regions)]

                # ── Draw each text part in its region ────────────────────────
                for r, t_part in zip(regions, text_parts):
                    if not t_part.strip():
                        continue
                    _draw_text_in_region(
                        draw, r["mask"], eroded_mask,
                        t_part, font_name, font_size, text_c,
                        padding, line_spacing,
                        test_only=False
                    )

            # Output empty bubble image
            empty_batch.append(_p2t_rgba(empty_bubble))

            # 5. Composite empty bubble onto original image
            if mask_blur > 0.0:
                r_ch, g_ch, b_ch, a_ch = empty_bubble.split()
                a_blurred = a_ch.filter(ImageFilter.GaussianBlur(mask_blur))
                empty_bubble_blurred = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_blurred))
            else:
                empty_bubble_blurred = empty_bubble

            # Resize bubble to match original image dimensions if they differ
            if empty_bubble_blurred.size != orig_pil.size:
                empty_bubble_blurred = empty_bubble_blurred.resize(orig_pil.size, Image.LANCZOS)

            comp_pil = Image.alpha_composite(orig_pil.convert("RGBA"), empty_bubble_blurred).convert("RGB")
            comp_batch.append(_p2t_rgb(comp_pil))

        return (
            torch.cat(comp_batch, dim=0),
            torch.cat(empty_batch, dim=0),
            torch.cat(masked_batch, dim=0),
        )

