"""面向 DeepFashion2 的快速端到端测试脚本。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, cast

import torch
from PIL import Image

from src.captioners.fashion_captioner import FashionCaptioner, FashionCaptionerConfig
from src.datasets.deepfashion2 import DeepFashion2Dataset
from src.detectors.grounding_dino_extractor import (
    PartDetection,
    GroundingDINOConfig,
    GroundingDINOExtractor,
)

STANDARD_PROMPTS = ["sleeve", "neckline", "hem", "waist"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fashion captioning smoke test")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="DeepFashion2 数据根目录，例如 /data/DeepFashion2",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["train", "val"],
        help="数据划分，默认为 val",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="测试样本数量上限",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/test_run.log",
        help="测试日志保存路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="显式指定推理设备，不提供则自动检测",
    )
    return parser.parse_args()


def resolve_device(forced: str | None) -> str:
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_logs(log_path: Path, lines: List[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(f"\n# Run at {timestamp}\n")
        for line in lines:
            fp.write(line + "\n")


def build_detected_dict(parts: Sequence[PartDetection]) -> Dict[str, List[Image.Image]]:
    buckets: Dict[str, List[Image.Image]] = {}
    for det in parts:
        buckets.setdefault(det.label, []).append(det.crop)
    return buckets


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    dataset = DeepFashion2Dataset(root_dir=args.root_dir, mode=args.mode)
    extractor = GroundingDINOExtractor(
        GroundingDINOConfig(device=device)
    )
    captioner = FashionCaptioner(FashionCaptionerConfig(device=device))

    log_lines: List[str] = []
    total = min(len(dataset), args.max_samples)
    for idx in range(total):
        image, target = dataset[idx]
        detections = extractor.detect_parts(image, STANDARD_PROMPTS)
        detected_dict = build_detected_dict(detections)
        caption = captioner.generate_description(
            image,
            cast(Dict[str, Sequence[Image.Image]], detected_dict),
        )
        line = (
            f"[{idx}] GT={target['category_name']} | Parts={list(detected_dict.keys())}"
            f" | Caption={caption}"
        )
        print(line)
        log_lines.append(line)

    save_logs(Path(args.log_path), log_lines)


if __name__ == "__main__":
    main()
