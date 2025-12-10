"""Formatting utilities for LLaVA instruction tuning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from PIL import Image


@dataclass(frozen=True)
class FormatterConfig:
    system_prompt: str = (
        "You are a senior fashion copywriter specialized in e-commerce storytelling."
    )
    user_template: str = (
        "<image>\n分析这张服装图片。我检测到了以下细节部位：{parts}。"
        "请结合这些细节，生成一段专业的电商营销文案。"
    )


class LLaVAInstructionFormatter:
    """Prepare conversation samples for LLaVA fine-tuning."""

    def __init__(self, config: FormatterConfig | None = None) -> None:
        self.config = config or FormatterConfig()

    def format_training_sample(
        self,
        image: Image.Image,
        caption: str,
        detected_parts: Sequence[str],
    ) -> Dict[str, Any]:
        parts_text = self._format_parts(detected_parts)
        user_prompt = self.config.user_template.format(parts=parts_text)

        conversation = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": caption.strip()},
        ]
        return {
            "images": [image],
            "conversations": conversation,
        }

    @staticmethod
    def _format_parts(parts: Sequence[str]) -> str:
        if not parts:
            return "未检测到显著细节"
        unique = []
        for part in parts:
            normalized = part.strip()
            if not normalized:
                continue
            if normalized not in unique:
                unique.append(normalized)
        return ", ".join(unique)
