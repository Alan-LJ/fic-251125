"""Grounding DINO 封装用于提取细节部位。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass
class VisualDetailConfig:
    model_name: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.35
    device: str | None = None
    torch_dtype: torch.dtype | None = None


class VisualDetailExtractor:
    def __init__(self, config: VisualDetailConfig | None = None) -> None:
        self.config = config or VisualDetailConfig()
        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = self.config.torch_dtype

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        model_kwargs: dict[str, object] = {"low_cpu_mem_usage": True}
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype

    @torch.inference_mode()
    def extract_details(
        self,
        image: Image.Image,
        category_list: Sequence[str],
    ) -> Dict[str, List]:
        if not category_list:
            return {"detection_results": [], "cropped_images": []}

        inputs = self.processor(
            images=image,
            text=", ".join(category_list),
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)

        outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            [image.size[::-1]],
            self.config.box_threshold,
            self.config.box_threshold,
        )[0]

        results: List[Dict] = []
        crops: List[Image.Image] = []
        for score, label, box in zip(
            processed["scores"], processed["labels"], processed["boxes"]
        ):
            score_val = float(score)
            if score_val < self.config.box_threshold:
                continue
            decoded_label = self.processor.tokenizer.decode(int(label))
            box_coords = self._clip_box(box, image.size)
            crop = image.crop(box_coords)
            results.append(
                {
                    "label": decoded_label.strip(),
                    "score": score_val,
                    "bbox": box_coords,
                }
            )
            crops.append(crop)

        return {"detection_results": results, "cropped_images": crops}

    @staticmethod
    def _clip_box(box_tensor, image_size):
        width, height = image_size
        x1 = max(int(box_tensor[0]), 0)
        y1 = max(int(box_tensor[1]), 0)
        x2 = min(int(box_tensor[2]), width)
        y2 = min(int(box_tensor[3]), height)
        return (x1, y1, x2, y2)

    def _move_inputs(self, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        moved: Dict[str, torch.Tensor] = {}
        for key, tensor in batch_inputs.items():
            if isinstance(tensor, torch.Tensor):
                target_dtype = self.model_dtype if tensor.is_floating_point() else None
                moved[key] = tensor.to(self.device, dtype=target_dtype)
            else:
                moved[key] = tensor
        return moved
