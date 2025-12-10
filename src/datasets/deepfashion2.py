"""DeepFashion2 数据集的 PyTorch Dataset 封装。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class InstanceAnnotation:
    """表示图像中单个服饰实例的标注。"""

    category: str
    bbox_xyxy: BBox


class DeepFashion2Dataset(Dataset):
    """按官方目录结构读取 DeepFashion2 图像及 JSON 标注。"""

    def __init__(
        self,
        root_dir: str | Path,
        mode: Literal["train", "val"],
        transforms: Optional[Any] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transforms = transforms

        self.image_dir, self.annotation_dir = self._resolve_split_dirs()

        self.image_paths: List[Path] = sorted(self.image_dir.glob("*.jpg"))
        if not self.image_paths:
            raise RuntimeError(f"在 {self.image_dir} 未发现 jpg 图像。")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        annotations = self._load_annotations(image_path)

        categories = [ann.category for ann in annotations]
        bboxes = [ann.bbox_xyxy for ann in annotations]

        target = {
            "categories": categories,
            "bboxes": bboxes,
        }

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def _load_annotations(self, image_path: Path) -> List[InstanceAnnotation]:
        json_path = self.annotation_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"缺少标注文件: {json_path}")

        with json_path.open("r", encoding="utf-8") as fp:
            payload: Dict[str, Any] = json.load(fp)

        items = self._extract_item_dicts(payload)
        if not items:
            raise ValueError(f"标注文件中未找到服饰实例: {json_path}")

        annotations: List[InstanceAnnotation] = []
        for item in items:
            raw_box = item.get("bounding_box")
            category = item.get("category_name")
            if raw_box is None or category is None:
                continue
            bbox = self._ensure_xyxy(raw_box)
            annotations.append(InstanceAnnotation(category=str(category), bbox_xyxy=bbox))

        if not annotations:
            raise ValueError(f"标注文件缺少有效 bbox/category: {json_path}")
        return annotations

    def _resolve_split_dirs(self) -> Tuple[Path, Path]:
        alias_map = {
            "train": ["train", "training"],
            "val": ["val", "validation", "val2019"],
        }
        candidate_roots: List[Path] = []
        for alias in alias_map.get(self.mode, [self.mode]):
            base = self.root_dir / alias
            candidate_roots.append(base)
            candidate_roots.append(base / alias)

        candidate_roots.append(self.root_dir / self.mode)
        candidate_roots.append(self.root_dir)

        image_names = ["image", "images", "img"]
        anno_names = ["json", "annos", "annotations"]

        for root in candidate_roots:
            for img_name in image_names:
                img_dir = root / img_name
                if not img_dir.exists():
                    continue
                for anno_name in anno_names:
                    anno_dir = root / anno_name
                    if anno_dir.exists():
                        return img_dir, anno_dir

        raise FileNotFoundError(
            "未找到包含 image/ 与 json/ (或 annos/) 目录的 DeepFashion2 切分，请检查 --root 与 --mode 参数。"
        )

    @staticmethod
    def _extract_item_dicts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """兼容 DeepFashion2 常见的 item 字段组织形式。"""

        if "items" in payload:
            data = payload["items"]
            if isinstance(data, list):
                return [obj for obj in data if isinstance(obj, dict)]
            if isinstance(data, dict):
                return [obj for obj in data.values() if isinstance(obj, dict)]

        item_dicts: List[Dict[str, Any]] = []
        for key, value in payload.items():
            if key.startswith("item") and isinstance(value, dict):
                item_dicts.append(value)
        return item_dicts

    @staticmethod
    def _ensure_xyxy(box: Sequence[float | int]) -> BBox:
        if len(box) != 4:
            raise ValueError("bounding_box 需要 4 个数值")
        x1, y1, w, h = (float(v) for v in box)
        return (x1, y1, x1 + max(w, 0.0), y1 + max(h, 0.0))

    def visualize_ground_truth(
        self,
        sample: Tuple[Image.Image, Dict[str, Sequence[BBox]]],
        window_name: str = "DeepFashion2 GT",
        wait: int = 0,
    ) -> np.ndarray:
        """在图像上绘制标注并使用 OpenCV 显示，返回带框数组。"""

        image, target = sample
        categories = target.get("categories", [])
        bboxes = target.get("bboxes", [])

        np_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for bbox, category in zip(bboxes, categories):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(np_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                np_img,
                str(category),
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, np_img)
        cv2.waitKey(wait)
        if wait >= 0:
            cv2.destroyWindow(window_name)
        return np_img
