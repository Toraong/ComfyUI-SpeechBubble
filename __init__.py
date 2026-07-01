"""
ComfyUI-SpeechBubble – __init__.py
Registers all custom nodes with ComfyUI.
"""

from .nodes import SpeechBubbleNode, SpeechBubbleSettingsNode, SpeechBubbleExtractorNode

NODE_CLASS_MAPPINGS = {
    "SpeechBubble":          SpeechBubbleNode,
    "SpeechBubbleSettings":  SpeechBubbleSettingsNode,
    "SpeechBubbleExtractor": SpeechBubbleExtractorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeechBubble":          "🗨️ Speech Bubble",
    "SpeechBubbleSettings":  "🗨️ Speech Bubble Settings",
    "SpeechBubbleExtractor": "🗨️ Speech Bubble Extractor & Replacer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
