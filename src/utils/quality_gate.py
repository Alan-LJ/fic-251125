"""质量过滤工具，利用 DeepFashion2 提供的部位分布信号。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from src.datasets.deepfashion2 import DeepFashion2Dataset


@dataclass(frozen=True)
class QualityDecision:
    keep: bool
    overlap: int
    required: int
    detected: int
    matched_parts: List[str]
    coverage: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "keep": self.keep,
            "overlap": self.overlap,
            "required": self.required,
            "detected": self.detected,
            "matched_parts": self.matched_parts,
            "coverage": self.coverage,
        }


class DeepFashion2QualityGate:
    """根据 DeepFashion2 样本中的类别统计，判断 DF-MM 样本是否保留。"""

    def __init__(
        self,
        root_dir: str | Path,
        mode: str = "val",
        max_reference_samples: Optional[int] = 2000,
        min_overlap: int = 1,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.max_reference_samples = max_reference_samples
        self.min_overlap = max(1, min_overlap)

        self.reference_counter: Counter[str] = Counter()
        self.allowed_parts: Set[str] = set()
        self._build_reference()

    def _normalize(self, name: str) -> str:
        return name.strip().lower()

    def _build_reference(self) -> None:
        dataset = DeepFashion2Dataset(self.root_dir, self.mode)
        limit = self.max_reference_samples or len(dataset)
        for idx in range(min(len(dataset), limit)):
            _, target = dataset[idx]
            for category in target.get("categories", []):
                normalized = self._normalize(category)
                if normalized:
                    self.reference_counter[normalized] += 1
        self.allowed_parts = {name for name, _ in self.reference_counter.items()}

    def evaluate(self, detected_parts: Sequence[str]) -> QualityDecision:
        normalized_detected = {
            self._normalize(part)
            for part in detected_parts
            if part and self._normalize(part)
        }
        matched = sorted(normalized_detected & self.allowed_parts)
        overlap = len(matched)
        coverage = (overlap / max(len(normalized_detected), 1)) if normalized_detected else 0.0
        keep = overlap >= self.min_overlap
        return QualityDecision(
            keep=keep,
            overlap=overlap,
            required=self.min_overlap,
            detected=len(normalized_detected),
            matched_parts=matched,
            coverage=coverage,
        )

    def summary(self, top_k: int = 20) -> Dict[str, object]:
        most_common = self.reference_counter.most_common(top_k)
        return {
            "mode": self.mode,
            "reference_size": sum(self.reference_counter.values()),
            "unique_parts": len(self.reference_counter),
            "top_parts": most_common,
        }
