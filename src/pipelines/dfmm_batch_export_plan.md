# DF-MM 批量导出方案

## 1. 目标
- 面向 DeepFashion-MultiModal（DF-MM）全量样本，批量生成 LLaVA 指令微调所需的 JSONL 记录。
- 复用 `VisualDetailExtractor`（Grounding DINO）输出的部位提示，避免重复算力开销，可缓存并断点续跑。
- 提供失败样本清单、处理指标统计，为后续 DeepFashion2 质量过滤与评测模块提供输入。

## 2. 输入 / 输出
### 输入
1. `--image-root`：DF-MM 图像根目录（复用 `DFMMDataset` 逻辑）。
2. `--annotation`：DF-MM captions.json / jsonl。
3. `--prompts`：Grounding DINO 细粒度提示，默认 `neckline,sleeve,waist,pattern`。
4. `--output`：导出 JSONL 路径；若存在则支持 `--resume` 追加。
5. `--cache-dir`：可选，保存中间产物（检测框、裁剪图、格式化文本）。
6. `--max-workers` / `--batch-size`：控制并发（I/O + 推理队列）。

### 输出 JSONL 字段建议
```json
{
  "image_id": "000123",
  "image_path": "images/000123.jpg",
  "gt_caption": "...",
  "prompts": ["neckline", "sleeve"],
  "dino": {
    "results": [{"label": "sleeve", "score": 0.58, "bbox": [..]}],
    "cached": true,
    "cache_path": "cache/dino/000/123.pt"
  },
  "llava_sample": {
    "conversation": [...],
    "has_parts": true
  },
  "status": "ok",
  "metrics": {
    "num_parts": 2,
    "avg_score": 0.54,
    "latency_ms": 320
  }
}
```
- 失败样本写入单独 JSONL（`--failure-log`），包含错误类型、堆栈摘要。

## 3. 模块拆分
1. `DFMMDataset`：负责索引与图像载入，按索引提供 `(PIL.Image, meta)`。
2. `DetectionCache`（新建）：
   - `get(image_id) -> Optional[DetectionPayload]`
   - `set(image_id, payload)`，payload 为 `VisualDetailExtractor.extract_details` 原样结构。
   - 默认使用 `cache/dino/{image_id[:3]}/{image_id}.pt`（torch.save dict）。
3. `DetailDetector`：包装 `VisualDetailExtractor`，内部先查缓存。
4. `SampleFormatter`：包装 `LLaVAInstructionFormatter`，输出 conversation 与额外统计（去重部位数、空部位标志）。
5. `ExporterWorker`：
   - 输入：`DFMMRecord`
   - 步骤：载入图像 → 检测 → 格式化 → 写输出队列。
   - 捕获异常，写入 failure 队列。
6. `JSONLWriter`：异步写入器，负责落盘主结果与失败文件，按队列消费，防止多线程竞争。
7. `ProgressTracker`：记录已处理 image_id 列表、统计成功/失败数、平均耗时；支持 `--resume` 时跳过已存在记录。

## 4. 处理流程
1. 初始化阶段：
   - 解析 CLI → 构造 `DFMMDataset`、`DetailDetector`、`SampleFormatter`。
   - 如果指定 `--resume`，读取已有输出 JSONL 中的 `image_id` 集合，构建跳过列表。
2. 主循环：
   - 使用 `DataLoader` 或 `range(len(dataset))` + `ThreadPoolExecutor`（I/O 可并行，DINO 推理仍在单 GPU 串行队列中）。
   - 对每个 record：
     1. 若 `image_id` 在 `processed_set` 中则 continue。
     2. 调 `DetailDetector.detect(image, prompts)` → 返回 detection dict（含缓存标志）。
     3. 提取 `detected_parts` 列表（score 过滤后）。
     4. 调 `SampleFormatter.format_training_sample(image, caption, detected_parts)`。
     5. 组装输出结构，推送至 `JSONLWriter`。
3. 写出阶段：
   - `JSONLWriter` 将 dict 序列化为单行 JSON，实时 flush，保证断电最小损失。
   - 同步更新 `ProgressTracker`（成功 +1、累计耗时）。
4. 结束：
   - 打印统计：总样本、成功、失败、检测缓存命中率、平均部位数量。
   - 生成 `stats.yaml`（或 json）供评测模块消费。

## 5. 断点续跑 & 缓存策略
- `--resume` 打开时：读取 `output_jsonl`，解析 image_id（可按行 `json.loads`）构成 `processed_set`。
- 检测缓存：
  - 命中：直接加载 `torch.load`，跳过 GPU 推理，写入 `"cached": true`。
  - 未命中：执行推理后 `torch.save` 至 cache 路径。
- 可选 `--force-redetect` 跳过缓存以更新模型。
- 失败重试：
  - 记录 `failure_log`，包含 `image_id`, `error`, `trace`。
  - 提供 `--retry-failures failure.jsonl` 从失败列表重跑。

## 6. 监控与日志
- 日志模块使用 `logging`，INFO 级别输出每 N 条进度，DEBUG 输出详细提示。
- 指标：
  - `processing_fps = processed / elapsed_time`。
  - `avg_detect_time`, `avg_format_time`（通过 `time.perf_counter`）。
  - 写入 `stats.yaml`。
- 提供 `--progress-interval` 控制控制台刷新频率。

## 7. CLI 草案
```bash
python -m src.pipelines.dfmm_batch_exporter \
  --image-root /data/dfmm/images \
  --annotation /data/dfmm/captions.json \
  --output outputs/dfmm_llava.jsonl \
  --cache-dir cache/dino \
  --prompts neckline sleeve waist pattern \ 
  --max-workers 4 --resume
```
- 模块入口 `src/pipelines/dfmm_batch_exporter.py`，封装 `main()`。

## 8. 与后续 TODO 的衔接
1. **实现批量导出管线**：依据上述模块拆分直接编码，逐步将 `ExporterWorker` + `JSONLWriter` 落地。
2. **集成 DF2 质量过滤**：导出后可以对 `metrics.num_parts` 或 `avg_score` 过低的样本打 `status=filtered_by_df2`，并在第二阶段读取 DeepFashion2 的部位频次提供更强约束。
3. **评测与配置**：`stats.yaml` 与 CLI 默认值可写入 `configs/dfmm_export.yaml`，便于复现。
