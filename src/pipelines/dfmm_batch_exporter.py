"""DeepFashion-MultiModal 批量导出 LLaVA 训练样本脚本。"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from PIL import Image

from src.datasets.dfmm import DFMMDataset
from src.detectors.visual_detail_extractor import VisualDetailExtractor
from src.utils.llava_formatter import LLaVAInstructionFormatter
from src.utils.quality_gate import DeepFashion2QualityGate


DEFAULT_PROMPTS = ["neckline", "sleeve", "waist", "pattern"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DF-MM 批量导出 JSONL")
    parser.add_argument("--image-root", required=True, help="DF-MM 图片根目录")
    parser.add_argument("--annotation", required=True, help="DF-MM 标注 (json/jsonl)")
    parser.add_argument("--output", required=True, help="导出 JSONL 文件")
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=DEFAULT_PROMPTS,
        help="Grounding DINO 细粒度提示词",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="可选，检测结果缓存目录",
    )
    parser.add_argument(
        "--failure-log",
        type=str,
        default=None,
        help="失败样本 JSONL，默认与输出同名 .fail.jsonl",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="运行统计 JSON，默认与输出同名 .stats.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若输出存在则跳过已处理 image_id 并追加写入",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="仅处理前 N 条样本（调试用）",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="每处理多少条样本打印一次进度",
    )
    parser.add_argument(
        "--df2-root",
        type=str,
        default=None,
        help="可选，提供 DeepFashion2 根目录以启用质量过滤",
    )
    parser.add_argument(
        "--df2-mode",
        type=str,
        default="val",
        choices=["train", "val"],
        help="质量过滤参考使用的 DeepFashion2 划分",
    )
    parser.add_argument(
        "--df2-max-reference",
        type=int,
        default=1000,
        help="取多少条 DeepFashion2 样本构建参考，默认 1000",
    )
    parser.add_argument(
        "--min-df2-overlap",
        type=int,
        default=1,
        help="保留 DF-MM 样本所需的 DF2 部位重叠数量",
    )
    parser.add_argument(
        "--drop-filtered",
        action="store_true",
        help="若 DF2 质量过滤不通过，则不写入主输出",
    )
    return parser.parse_args()


class DetectionCache:
    """简单的 JSON 缓存：仅保存 detection_results，避免重复推理。"""

    def __init__(self, cache_dir: str | Path | None) -> None:
        self.base = Path(cache_dir) if cache_dir else None
        if self.base:
            self.base.mkdir(parents=True, exist_ok=True)

    def _path_for(self, image_id: str) -> Optional[Path]:
        if not self.base:
            return None
        bucket = image_id[:3] or "unk"
        return self.base / bucket / f"{image_id}.json"

    def get(self, image_id: str) -> Optional[Dict]:
        path = self._path_for(image_id)
        if not path or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            return payload
        except json.JSONDecodeError:
            logging.warning("缓存损坏: %s", path)
            return None

    def set(self, image_id: str, detection_payload: Dict) -> None:
        path = self._path_for(image_id)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(detection_payload, fp, ensure_ascii=False)


@dataclass
class ExportStats:
    processed: int = 0
    failed: int = 0
    filtered: int = 0
    cache_hits: int = 0
    detection_time: float = 0.0
    total_parts: int = 0
    total_samples: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record_success(self, cached: bool, detection_seconds: float, num_parts: int) -> None:
        self.processed += 1
        self.total_samples += 1
        self.total_parts += num_parts
        self.detection_time += detection_seconds
        if cached:
            self.cache_hits += 1

    def record_failure(self) -> None:
        self.failed += 1
        self.total_samples += 1

    def record_filtered(self) -> None:
        self.filtered += 1
        self.total_samples += 1

    def to_dict(self) -> Dict[str, float | int]:
        elapsed = max(time.perf_counter() - self.start_time, 1e-6)
        avg_detection = self.detection_time / self.processed if self.processed else 0.0
        avg_parts = self.total_parts / self.processed if self.processed else 0.0
        cache_hit_rate = self.cache_hits / self.processed if self.processed else 0.0
        return {
            "total_samples": self.total_samples,
            "processed": self.processed,
            "failed": self.failed,
            "filtered": self.filtered,
            "cache_hit_rate": cache_hit_rate,
            "avg_detection_ms": avg_detection * 1000.0,
            "avg_parts_per_sample": avg_parts,
            "throughput_samples_per_min": self.processed / elapsed * 60.0,
        }


def load_processed_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    processed: Set[str] = set()
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                image_id = obj.get("image_id")
                if image_id:
                    processed.add(str(image_id))
            except json.JSONDecodeError:
                continue
    logging.info("检测到 %d 条已处理记录，将在 resume 模式下跳过", len(processed))
    return processed


def detect_with_cache(
    extractor: VisualDetailExtractor,
    cache: DetectionCache,
    image_id: str,
    image: Image.Image,
    prompts: Sequence[str],
) -> tuple[Dict, bool]:
    cached_payload = cache.get(image_id)
    if cached_payload is not None:
        return cached_payload, True

    detection = extractor.extract_details(image, prompts)
    payload = {
        "detection_results": detection.get("detection_results", []),
    }
    cache.set(image_id, payload)
    return payload, False


def ensure_failure_log_path(output_path: Path, failure_path: str | None) -> Path:
    if failure_path:
        return Path(failure_path)
    return output_path.with_suffix(".fail.jsonl")


def ensure_stats_path(output_path: Path, stats_path: str | None) -> Path:
    if stats_path:
        return Path(stats_path)
    return output_path.with_suffix(".stats.json")


def write_jsonl(fp, payload: Dict) -> None:
    fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    fp.flush()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    output_path = Path(args.output)
    failure_path = ensure_failure_log_path(output_path, args.failure_log)
    stats_path = ensure_stats_path(output_path, args.stats)

    if output_path.exists() and not args.resume:
        logging.warning("输出文件已存在，将覆盖: %s", output_path)

    dataset = DFMMDataset(args.image_root, args.annotation)
    extractor = VisualDetailExtractor()
    formatter = LLaVAInstructionFormatter()
    cache = DetectionCache(args.cache_dir)
    quality_gate = (
        DeepFashion2QualityGate(
            root_dir=args.df2_root,
            mode=args.df2_mode,
            max_reference_samples=args.df2_max_reference,
            min_overlap=args.min_df2_overlap,
        )
        if args.df2_root
        else None
    )
    if quality_gate:
        logging.info(
            "启用 DF2 质量过滤：参考样本 %s，最小重叠 %d",
            quality_gate.summary().get("reference_size"),
            args.min_df2_overlap,
        )

    processed_ids = load_processed_ids(output_path) if args.resume else set()
    stats = ExportStats()
    prompts = args.prompts or DEFAULT_PROMPTS

    attempted = 0
    max_samples = args.max_samples

    with output_path.open("a" if args.resume else "w", encoding="utf-8") as out_fp, \
        failure_path.open("a" if args.resume else "w", encoding="utf-8") as fail_fp:

        for idx, record in enumerate(dataset.records):
            if max_samples is not None and attempted >= max_samples:
                break

            image_id = record.image_id
            if image_id in processed_ids:
                continue

            attempted += 1

            try:
                image, meta = dataset[idx]
                start = time.perf_counter()
                detection_payload, from_cache = detect_with_cache(
                    extractor, cache, image_id, image, prompts
                )
                detect_time = time.perf_counter() - start

                detected_parts = [
                    res.get("label", "")
                    for res in detection_payload.get("detection_results", [])
                    if res.get("score", 0.0) >= extractor.config.box_threshold
                ]

                quality_payload = None
                status = "ok"
                if quality_gate:
                    decision = quality_gate.evaluate(detected_parts)
                    quality_payload = decision.to_dict()
                    if not decision.keep:
                        status = "filtered_df2"
                        if args.drop_filtered:
                            stats.record_filtered()
                            failure_entry = {
                                "image_id": image_id,
                                "error": "FilteredByDF2",
                                "message": "未满足 DF2 重叠阈值",
                                "quality": quality_payload,
                            }
                            write_jsonl(fail_fp, failure_entry)
                            continue

                formatted = formatter.format_training_sample(
                    image,
                    meta["caption"],
                    detected_parts,
                )

                entry = {
                    "image_id": image_id,
                    "image_path": str(record.image_path),
                    "gt_caption": meta["caption"],
                    "prompts": list(prompts),
                    "dino": {
                        "results": detection_payload.get("detection_results", []),
                        "cached": from_cache,
                    },
                    "llava_sample": {
                        "conversation": formatted["conversations"],
                        "has_parts": bool(detected_parts),
                    },
                    "status": status,
                }
                if quality_payload:
                    entry["quality"] = quality_payload

                write_jsonl(out_fp, entry)
                stats.record_success(from_cache, detect_time, len(detected_parts))

                if stats.processed % args.progress_every == 0:
                    logging.info(
                        "已处理 %d 条，缓存命中率 %.2f%%",
                        stats.processed,
                        stats.cache_hits / max(stats.processed, 1) * 100,
                    )

            except Exception as exc:  # noqa: BLE001 广义捕获以保证批处理不中断
                stats.record_failure()
                failure_entry = {
                    "image_id": image_id,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
                write_jsonl(fail_fp, failure_entry)
                logging.exception("处理 %s 失败", image_id)

    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats.to_dict(), fp, ensure_ascii=False, indent=2)
    logging.info("导出完成：成功 %d，失败 %d", stats.processed, stats.failed)


if __name__ == "__main__":
    main()