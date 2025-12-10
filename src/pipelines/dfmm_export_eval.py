"""读取 DF-MM 导出 JSONL 生成覆盖率/质量报告。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("统计 DF-MM 导出结果")
    parser.add_argument("--input", required=True, help="导出 JSONL 文件")
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="可选，将指标写入 JSON 报告路径",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="输出出现频率最高的 K 个部位",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="统计部位时的最小置信分数阈值",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize(part: str) -> str:
    return part.strip().lower()


def summarize(records: Iterable[Dict], min_score: float, top_k: int) -> Dict[str, object]:
    total = 0
    status_counter: Counter[str] = Counter()
    part_counter: Counter[str] = Counter()
    parts_per_sample: List[int] = []
    coverage_flags: List[int] = []
    quality_kept = 0
    quality_total = 0
    quality_coverages: List[float] = []

    for entry in records:
        total += 1
        status = entry.get("status", "ok")
        status_counter[status] += 1

        results = entry.get("dino", {}).get("results", [])
        filtered = [
            res
            for res in results
            if res.get("score", 0.0) >= min_score and res.get("label")
        ]
        unique_parts = {normalize(str(res["label"])) for res in filtered}
        unique_parts.discard("")
        part_counter.update(unique_parts)
        parts_per_sample.append(len(unique_parts))
        coverage_flags.append(1 if unique_parts else 0)

        quality = entry.get("quality")
        if isinstance(quality, dict):
            quality_total += 1
            if quality.get("keep"):
                quality_kept += 1
            coverage_val = float(quality.get("coverage", 0.0))
            quality_coverages.append(coverage_val)

    def safe_mean(values: List[float]) -> float:
        return mean(values) if values else 0.0

    report = {
        "total_records": total,
        "status_breakdown": dict(status_counter),
        "avg_parts_per_sample": safe_mean([float(v) for v in parts_per_sample]),
        "part_coverage_rate": safe_mean([float(v) for v in coverage_flags]),
        "top_parts": part_counter.most_common(top_k),
        "quality_gate": {
            "evaluated": quality_total,
            "kept": quality_kept,
            "keep_rate": (quality_kept / quality_total) if quality_total else 0.0,
            "avg_quality_coverage": safe_mean(quality_coverages),
        },
    }
    return report


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    report = summarize(iter_jsonl(input_path), args.min_score, args.top_k)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()