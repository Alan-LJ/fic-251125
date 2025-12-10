"""验证 DeepFashion2 数据加载 + Grounding DINO + LLaVA 格式化流程。"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import List

from src.datasets.deepfashion2 import DeepFashion2Dataset
from src.detectors.visual_detail_extractor import VisualDetailExtractor
from src.utils.llava_formatter import LLaVAInstructionFormatter


DEFAULT_PROMPTS = ["neckline", "sleeve", "waist", "pattern"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Verify DeepFashion2 + detector + formatter")
    parser.add_argument("--root", required=True, help="DeepFashion2 数据集根目录")
    parser.add_argument(
        "--mode",
        choices=("train", "val"),
        default="val",
        help="使用的划分 (train/val)",
    )
    parser.add_argument("--index", type=int, default=0, help="样本索引")
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=DEFAULT_PROMPTS,
        help="传给 Grounding DINO 的细粒度提示",
    )
    return parser.parse_args()


def build_reference_caption(categories: List[str]) -> str:
    if not categories:
        return "Ground truth annotations are missing for this sample."
    counts = Counter(categories)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = [f"{name} x{count}" for name, count in ordered]
    return "Ground truth categories: " + ", ".join(parts)


def main() -> None:
    args = parse_args()

    dataset = DeepFashion2Dataset(args.root, args.mode)
    extractor = VisualDetailExtractor()
    formatter = LLaVAInstructionFormatter()

    image, target = dataset[args.index]
    gt_categories = target.get("categories", [])
    gt_bboxes = target.get("bboxes", [])

    detection = extractor.extract_details(image, args.prompts)
    detected_parts = [res["label"] for res in detection["detection_results"]]

    caption = build_reference_caption(gt_categories)
    formatted = formatter.format_training_sample(image, caption, detected_parts)
    user_prompt = next(
        (turn["content"] for turn in formatted["conversations"] if turn["role"] == "user"),
        "",
    )

    image_path = dataset.image_paths[args.index]
    print(f"Image Path: {image_path}")
    print(f"Ground Truth Categories ({len(gt_categories)}): {gt_categories}")
    print(f"Ground Truth BBoxes ({len(gt_bboxes)}): {gt_bboxes[:3]}{' ...' if len(gt_bboxes) > 3 else ''}")
    print(f"Detected Parts ({len(detected_parts)}): {detected_parts}")
    print("Constructed User Prompt:\n" + user_prompt)
    print(f"Reference Caption: {caption}")


if __name__ == "__main__":
    main()
