"""基于 LLaVA 的服饰描述生成器。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


@dataclass(frozen=True)
class FashionCaptionerConfig:
    """管理 LLaVA 模型推理所需的核心参数。"""

    model_name: str = "llava-hf/llava-1.5-7b-hf"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95


class FashionCaptioner:
    """利用 LLaVA 模型生成带细粒度部位提示的服饰文案。"""

    def __init__(self, config: FashionCaptionerConfig | None = None) -> None:
        self.config = config or FashionCaptionerConfig()
        self._processor: Any | None = None
        self._model: Any | None = None

    def load(self) -> None:
        """懒加载模型，避免重复初始化。"""

        if self._processor is not None and self._model is not None:
            return
        self._processor = AutoProcessor.from_pretrained(self.config.model_name)
        model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.dtype,
            low_cpu_mem_usage=True,
            ),
        )
        model.to(self.config.device)
        model.eval()
        self._model = model

    @torch.inference_mode()
    def generate_description(
        self,
        original_image: Image.Image,
        detected_crops_dict: Optional[Dict[str, Sequence[Image.Image]]] = None,
    ) -> str:
        """构造部位提示并生成描述文本。"""

        self.load()
        assert self._processor is not None and self._model is not None

        part_names = self._extract_part_names(detected_crops_dict)
        if part_names:
            parts_str = ", ".join(part_names)
            prompt = (
                "Describe this clothing item in detail. Focus on these detected parts: "
                f"{parts_str}."
            )
        else:
            prompt = "Describe this clothing item in detail."

        chat = [
            {
                "role": "system",
                "content": "You are a creative fashion copywriter who highlights fine-grained garment details.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]

        processor = self._processor
        model = self._model

        conversation = processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.config.device)

        inputs = processor(
            images=original_image,
            text=conversation,
            return_tensors="pt",
        ).to(self.config.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()

    @staticmethod
    def _extract_part_names(
        detected_crops_dict: Optional[Dict[str, Sequence[Image.Image]]]
    ) -> List[str]:
        if not detected_crops_dict:
            return []
        unique_parts: List[str] = []
        for part_name in detected_crops_dict.keys():
            normalized = part_name.strip().lower()
            if normalized and normalized not in unique_parts:
                unique_parts.append(normalized)
        return unique_parts
