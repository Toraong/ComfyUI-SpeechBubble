"""
ComfyUI-SpeechBubble – __init__.py
Registers all custom nodes with ComfyUI.
"""

from .nodes import SpeechBubbleNode, SpeechBubbleSettingsNode

NODE_CLASS_MAPPINGS = {
    "SpeechBubble":         SpeechBubbleNode,
    "SpeechBubbleSettings": SpeechBubbleSettingsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeechBubble":         "🗨️ Speech Bubble",
    "SpeechBubbleSettings": "🗨️ Speech Bubble Settings",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
