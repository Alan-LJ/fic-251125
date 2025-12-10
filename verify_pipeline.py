"""Quick verification script to connect DF-MM dataset, DINO, and formatter."""
from __future__ import annotations

import argparse
from typing import List

from PIL import Image

from src.datasets.dfmm import DFMMDataset
from src.detectors.visual_detail_extractor import VisualDetailExtractor
from src.utils.llava_formatter import LLaVAInstructionFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify formatting pipeline")
    parser.add_argument("--image-root", type=str, required=True, help="DF-MM image root")
    parser.add_argument("--annotation", type=str, required=True, help="DF-MM annotation file")
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=["neckline", "sleeve", "waist", "pattern"],
        help="Fine-grained prompts for DINO",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = DFMMDataset(args.image_root, args.annotation)
    extractor = VisualDetailExtractor()
    formatter = LLaVAInstructionFormatter()

    image, meta = dataset[args.index]
    caption = meta["caption"]
    image_id = meta["image_id"]

    detection = extractor.extract_details(image, args.prompts)
    detected_parts = [res["label"] for res in detection["detection_results"]]

    formatted = formatter.format_training_sample(image, caption, detected_parts)
    conversation = formatted["conversations"]
    user_prompt = next((turn["content"] for turn in conversation if turn["role"] == "user"), "")

    print(f"Image ID: {image_id}")
    print(f"Input Image Shape: {image.size}")
    print(f"Detected Parts: {detected_parts}")
    print("Constructed User Prompt:\n" + user_prompt)
    print(f"Target Caption: {caption}")


if __name__ == "__main__":
    main()
