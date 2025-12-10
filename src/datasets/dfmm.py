"""DeepFashion-MultiModal (DF-MM) 数据集封装。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DFMMRecord:
    image_id: str
    image_path: Path
    caption: str


class DFMMDataset(Dataset):
    """支持 JSON / JSONL 注释格式的 DF-MM 数据读取。"""

    def __init__(self, image_root: str | Path, annotation_file: str | Path) -> None:
        self.image_root = Path(image_root)
        self.annotation_file = Path(annotation_file)
        if not self.image_root.exists():
            raise FileNotFoundError(f"未找到图片根目录: {self.image_root}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"未找到标注文件: {self.annotation_file}")

        self.records = self._load_records()
        if not self.records:
            raise RuntimeError("DF-MM 标注文件为空或解析失败")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            image = Image.open(record.image_path).convert("RGB")
        except FileNotFoundError:
            print(f"[警告] 图像缺失: {record.image_path}")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        caption = record.caption
        return image, {"caption": caption, "image_id": record.image_id}

    def _load_records(self) -> List[DFMMRecord]:
        suffix = self.annotation_file.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl()
        if suffix == ".json":
            return self._load_json()
        raise ValueError(f"不支持的标注文件格式: {self.annotation_file}")

    def _load_json(self) -> List[DFMMRecord]:
        with self.annotation_file.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)

        if isinstance(payload, dict):
            iterable = (
                {"image_id": key, "image": key, "caption": value}
                for key, value in payload.items()
            )
        elif isinstance(payload, list):
            iterable = payload
        else:
            raise ValueError("JSON 标注需为列表或字典")

        records: List[DFMMRecord] = []
        for entry in iterable:
            record = self._parse_entry(entry)
            if record:
                records.append(record)
        return records

    def _load_jsonl(self) -> List[DFMMRecord]:
        records: List[DFMMRecord] = []
        with self.annotation_file.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                record = self._parse_entry(entry)
                if record:
                    records.append(record)
        return records

    def _parse_entry(self, entry) -> DFMMRecord | None:
        image_id = entry.get("image_id") or entry.get("id")
        caption = entry.get("caption") or entry.get("text")
        rel_path = entry.get("image") or entry.get("image_path")
        if not image_id or not caption or not rel_path:
            return None
        image_path = self.image_root / rel_path
        return DFMMRecord(image_id=str(image_id), image_path=image_path, caption=str(caption))
