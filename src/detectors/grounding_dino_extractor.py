"""Grounding DINO 裁剪器封装，便于提取服饰部位图块。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from transformers import AutoModelForObjectDetection, AutoProcessor


@dataclass(frozen=True)
class GroundingDINOConfig:
    """管理模型名称与推理阈值等超参。"""

    model_name: str = "IDEA-Research/grounding-dino-base"
    device: str = "cuda"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    max_detections: int = 8


@dataclass
class PartDetection:
    """存储裁剪图块及其标签与分数。"""

    label: str
    score: float
    crop: Image.Image


class GroundingDINOExtractor:
    """包装 Grounding DINO 以返回服饰部位裁剪。"""

    def __init__(self, config: GroundingDINOConfig | None = None) -> None:
        self.config = config or GroundingDINOConfig()
        self._processor = None
        self._model = None

    def load(self) -> None:
        """懒加载模型与处理器，避免重复开销。"""

        if self._processor is not None and self._model is not None:
            return
        self._processor = AutoProcessor.from_pretrained(self.config.model_name)
        self._model = AutoModelForObjectDetection.from_pretrained(self.config.model_name)
        self._model.to(self.config.device)
        self._model.eval()

    @torch.inference_mode()
    def detect_parts(self, image: Image.Image, text_prompts: Sequence[str]) -> List[PartDetection]:
        """根据多条提示词返回裁剪图块与分数。"""

        if not text_prompts:
            return []
        self.load()
        assert self._processor is not None and self._model is not None

        results: List[PartDetection] = []
        target_sizes = [image.size[::-1]]  # height, width for post-process scaling

        # 逐个提示词检测，便于显式控制标签映射。
        for prompt in text_prompts:
            phrased_prompt = f"find {prompt}"  # 引导模型聚焦具体部位
            inputs = self._processor(
                images=image,
                text=phrased_prompt,
                return_tensors="pt",
            ).to(self.config.device)

            outputs = self._model(**inputs)
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                target_sizes=target_sizes,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
            )[0]

            if len(processed["scores"]) == 0:
                continue

            for score, box in zip(processed["scores"], processed["boxes"]):
                score_val = float(score)
                if score_val < self.config.box_threshold:
                    continue
                x_min, y_min, x_max, y_max = self._clip_box(box, image.size)
                crop = image.crop((x_min, y_min, x_max, y_max))
                results.append(PartDetection(label=prompt, score=score_val, crop=crop))

        results.sort(key=lambda det: det.score, reverse=True)
        return results[: self.config.max_detections]

    @staticmethod
    def _clip_box(box_tensor, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        """将检测框裁剪到图像范围，并转换为整数像素坐标。"""

        width, height = image_size
        x_min = max(int(box_tensor[0]), 0)
        y_min = max(int(box_tensor[1]), 0)
        x_max = min(int(box_tensor[2]), width)
        y_max = min(int(box_tensor[3]), height)
        return x_min, y_min, x_max, y_max
