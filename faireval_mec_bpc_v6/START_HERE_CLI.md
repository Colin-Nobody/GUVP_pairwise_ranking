# 无记忆 CLI 交接入口

如果你是刚打开本仓库、没有此前聊天记录的 CLI/开发者，请先完整阅读本文件，再阅读 [`EUROPE_50_CITIES_5090_PLAN.md`](EUROPE_50_CITIES_5090_PLAN.md) 和 [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md)。不要仅凭文件名重跑实验。

## 项目目标

本目录是 `GUVP_pairwise_ranking` 中相互隔离的 FairEval MEC+BPC V6 实验包。目标是让 Qwen VLM 比较两张街景图片的视觉污染程度，并用：

- MEC：每个图像位置独立采样 3 次证据与分数；
- BPC：同一无序图片对分别以 AB 和 BA 顺序判断；
- Q-score 与 TrueSkill：聚合有效的校准胜负票。

最终 VP 分数越高表示视觉污染越严重。正式 7B 模型是 `Qwen/Qwen2.5-VL-7B-Instruct`。

## 绝对不要擅自改变的控制变量

除非用户明确授权，不得修改以下内容：

```text
model = Qwen/Qwen2.5-VL-7B-Instruct
quantization = 4-bit NF4, double quantization, BF16 compute
max_pixels = 401408
MEC = 3 samples per position
BPC = AB + BA
temperature = 1.0
max_new_tokens = 192
seed = 42
label layout = label before image
prompt SHA-256 = b34d3f8b72f684ae228edc6ef817b27d07927d8a73b3a5bd172cfc8b2f33fa8e
```

尤其不要自行“修正”提示词。已有输出中偶尔会把正常行人、车辆、自行车或植被描述成污染，也可能出现推测性证据；用户已明确要求保留原提示词。应通过审计与人工复核记录这一风险，而不是静默修改语义。

## 当前实现状态

`vlm_pairwise.py` 已实现两条解码路径：

1. 正式路径：`--mec-batch-size 1 --reuse-prefix-cache`。串行保留各次 MEC 的独立采样，同时复用视觉/提示词 KV 前缀。
2. 实验路径：`--mec-batch-size 2/3`。吞吐可能更高，但会改变随机采样轨迹；不能直接和正式审计文件混跑。

截至 2026-09-01 的小规模对照结果：

- 前 20 对、120 个方向样本与旧版正式 7B 输出逐字一致；
- 校准分数、winner、tie、invalid 完全一致；
- 峰值 CUDA allocated 约 5.88 GiB；
- 优化版小样本约 242 对/小时，原版完整实验约 202 对/小时；
- batch 3 小测约 262 对/小时，但 20 对解析样本从基线 113/120 降至 109/120，正式脚本因此不采用。

这些数字只是在 RTX 5060 8GB 上的历史测量，不应当冒充 RTX 5090 实测结果。详细记录见 [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md)。

## 文件地图

- [`vlm_pairwise.py`](vlm_pairwise.py)：MEC+BPC 推理、断点恢复、前缀缓存及实验批处理。
- [`MODEL_CONFIG.json`](MODEL_CONFIG.json)：机器可读控制变量。
- [`scripts/run_7b_formal_optimized_and_score.sh`](scripts/run_7b_formal_optimized_and_score.sh)：正式 7B 前缀缓存运行入口。
- [`scripts/monitor_7b_formal_optimized.sh`](scripts/monitor_7b_formal_optimized.sh)：正式运行监控。
- [`tests/test_vlm_pairwise.py`](tests/test_vlm_pairwise.py)：解析、校准、批采样与缓存单元测试。
- [`data/vlm_pairwise_formal_balanced24_v6_3000.csv`](data/vlm_pairwise_formal_balanced24_v6_3000.csv)：250 图、每图 24 次、共 3,000 对的固定清单。
- [`EUROPE_50_CITIES_5090_PLAN.md`](EUROPE_50_CITIES_5090_PLAN.md)：50 城市扩展方案、配对图、分片和工时公式。
- [`RESULTS.md`](RESULTS.md)：旧版 3,000 对正式结果与限制。

模型权重、街景原图、token、活动日志以及大型原始审计文件不在 Git 中。

## 首次接手检查

在本目录执行：

```bash
git status --short
python -m unittest discover -s tests -v
bash -n scripts/run_7b_formal_optimized_and_score.sh
bash -n scripts/monitor_7b_formal_optimized.sh
```

如果本机已有原图，再做输入预检：

```bash
python scripts/preflight_formal_abba.py \
  --images-dir data/street_images \
  --pairs data/vlm_pairwise_formal_balanced24_v6_3000.csv \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --expected-images 250 \
  --expected-pairs 3000 \
  --mec-samples 3
```

不要假设以前的本地进程仍在运行。先只读检查：

```bash
pgrep -af '[v]lm_pairwise.py' || true
find output -maxdepth 2 -type f -name 'progress.log' -print 2>/dev/null
```

发现活动推理进程时，不要另启模型进程争抢 GPU，也不要删除其审计文件。

## 250 图正式运行

从本目录启动：

```bash
./scripts/run_7b_formal_optimized_and_score.sh
```

脚本默认沿用原实验机的 `/home/colin/miniconda3/envs/guvp-vlm/bin/python`。在新机器上应显式指定当前环境，不要修改脚本来硬编码另一台机器：

```bash
PYTHON_BIN="$(command -v python)" ./scripts/run_7b_formal_optimized_and_score.sh
```

另一个终端监控：

```bash
./scripts/monitor_7b_formal_optimized.sh 10
```

脚本在 `pair_audit.csv` 已存在时自动传入 `--resume`。要做真正的从零实验，必须先确认没有活动进程，并将旧输出移动到有日期的归档目录；不要直接删除不可恢复的数据。

## 50 城市下一阶段的起点

推荐默认设计是 50 城市 × 每城 1,000 图，每图约 25 次：

```text
images = 50,000
unordered pairs = 50,000 × 25 / 2 = 625,000
model generations = 625,000 × 2 positions × 3 MEC = 3,750,000
```

每图推荐约 20 个同城对和 5 个跨城对，使全局比较图连通。完整构图、分片、验证门槛和 RTX 5090 工时见 [`EUROPE_50_CITIES_5090_PLAN.md`](EUROPE_50_CITIES_5090_PLAN.md)。

关键阻塞项：用户尚未在此仓库中提供最终 50 城市名单、图像来源与许可决策。未来 CLI 不得自行编造名单并当成已批准输入。

50 城市的元数据采集器、25 度连通配对生成器、分片运行器和合并器目前也尚未实现；文档描述的是下一阶段规格。实现它们时应新增自动测试，并先在小规模合成数据上证明度数、连通性、恢复和合并不变量。

## 正式扩容前的决策顺序

1. 取得用户确认的城市清单、边界、图像来源和许可。
2. 先以少量城市验证空间分层采样和元数据完整性。
3. 在目标 RTX 5090 上，用固定 30 对依次基准 batch 1、3、6、9/12；每个配置写入独立输出目录。
4. 比较吞吐、峰值显存、解析率、位置偏差、BPC 冲突与分数分布。
5. 只有通过方法等价性门槛后，才生成最终 625,000 对清单并分片正式运行。
6. 所有分片完成、去重并验证完整后，统一计算一次全局 Q-score/TrueSkill；不要分别对每片排名后再拼接。

在任何代码变更前，先说明它属于“严格复现优化”还是“统计方法等价优化”。不同 GPU 或不同 batching 下不应承诺逐字节相同随机输出。
