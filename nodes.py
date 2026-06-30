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
    lines: List[str] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        cur = ""
        for word in words:
            test = (cur + " " + word).strip()
            bb = draw.textbbox((0, 0), test, font=font)
            if bb[2] - bb[0] <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
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
                "bubble_x":    ("INT", {"default": 50,  "min": 0,     "max": 8192, "step": 1}),
                "bubble_y":    ("INT", {"default": 50,  "min": 0,     "max": 8192, "step": 1}),
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
                "bubble_x": ("INT", {"default": 50,  "min": 0,     "max": 8192}),
                "bubble_y": ("INT", {"default": 50,  "min": 0,     "max": 8192}),
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
