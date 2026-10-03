# GUVP Pairwise Ranking

完整实验归档、DGX Spark 迁移、环境恢复与正式 VLM 复现说明见
[`DGX_SPARK_MIGRATION.md`](DGX_SPARK_MIGRATION.md)。

将人工（后续也可由 VLM 产生）的街景两两比较结果，转换为每张图片的视觉污染（Visual Pollution, VP）分数。

本项目只实现两种既有的聚合方法：

- **Place Pulse Q-score**：以胜率为基础，使用对手的胜率与负率进行 strength-of-schedule 校正。
- **TrueSkill**：将每张图片视作一个选手，按每次两两比较更新后验评分；这是主结果。

`winner` 必须表示在本次比较的目标感知维度中**更受偏好的一张图片**。对于 VP，建议题目固定为：

> 哪张街景看起来更干净、视觉污染更低？

这样 TrueSkill 高分表示「更干净／更低视觉污染」，而输出的 VP 指数为其反向值：`VP = 1 - normalized_preference`。因此 VP 越高，表示视觉污染越严重。

## 安装与运行

```bash
cd GUVP_pairwise_ranking
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --input data/pairwise_votes.csv --output-dir output --dimension visual_pollution
```

运行后会生成：

- `output/perception_scores.csv`：每张图的 Q-score、TrueSkill、VP 指数与基础统计。
- `output/ranking_summary.json`：数据质量和 Q-score/TrueSkill 的 Pearson、Spearman 相关性。
- `output/score_distributions.png`：三种得分的直方图。
- `output/score_comparison.png`：Q-score 与 TrueSkill 偏好分的散点图。

## 输入 CSV

必需列为 `image_a,image_b,winner`。`winner` 必须等于同一行的 `image_a` 或 `image_b`。可选 `comparison_id`、`rater_id`、`dimension` 等列会保留在原始文件中，但不参与计算。

```csv
image_a,image_b,winner
img001,img002,img001
img001,img003,img003
img002,img003,img002
```

每一行对应一张问卷作答，而不是唯一图片对；同一图片对的多位受试者投票应保留为多行。图片 ID 可以是文件名、数据库 ID 或未来 VLM 任务 ID。

## 输出字段

- `q_score_0_10`：Place Pulse 风格 Q-score，0–10。
- `trueskill_mu` / `trueskill_sigma`：TrueSkill 后验均值与不确定性。
- `trueskill_preference_0_1`：将 µ 线性归一化的偏好分；高 = 更干净。
- `visual_pollution_score_0_1`：主 VP 指数；高 = 视觉污染更严重。
- `visual_pollution_score_0_10`：便于报告的 0–10 VP 指数。
- `wins`、`losses`、`comparison_count`、`win_ratio`：审计和数据质量字段。

当所有图片的 TrueSkill µ 相同（例如只有一张图片或没有有效比较）时，归一化偏好分固定为 0.5，以避免人为制造排序。

## 方法说明

Q-score 对图片 i 定义为：

`Q_i = 10 / 3 * (win_rate_i + mean(win_rate of opponents i beat) - mean(loss_rate of opponents that beat i) + 1)`。

这正是 Place Pulse 后续工作中所述的 strength-of-schedule 计算：击败强对手获得更多奖励，输给弱对手得到更多惩罚。TrueSkill 的参数与近期街景两两比较研究一致：`mu=25`、`sigma=25/3`、`beta=25/6`、`tau=25/300`、`draw_probability=0`。输入行顺序会影响 TrueSkill 的逐次后验更新；为实现可复现性，请保存问卷导出时的原始行顺序。

## 后续接入 VLM

无需改动算法。只要将 VLM 的每个强制二选一结果写成同一 CSV 模式即可。例如让 VLM 回答「A 和 B 哪一张更干净？」并将被选择的图片 ID 填入 `winner`。建议另存 VLM 名称、prompt 版本、温度、时间戳和置信度等元数据列，便于审计。

### 本地 Qwen VLM：MEC + BPC 校准

`vlm_pairwise.py` 默认使用 Qwen2.5-VL-7B、4-bit NF4 量化和前置 A/B 标签，并按 Wang et al. 的 FairEval 方法适配街景 VP：

- **MEC**：模型先写可见物证据，再给两图各自的 1–10 清洁度分数；每个位置用 temperature=1.0 独立采样 k=3 次。
- **BPC**：同一图片对同时运行 AB 与 BA。程序把交换后的分数重新映射到实际图片，对每张图的 2k 个分数取平均；平均分更高者才成为该对的 `winner`。
- 若任一位置没有得到完整 k 个可解析结果，或两个校准均分相同，该图片对只保留在 audit CSV，不进入 `main.py` 的 votes CSV。审计同时保存全部原始响应、位置分数、校准分差和 BPDE。

因此默认每个无序图片对需要 2×3=6 次推理。允许的正式模型仅为 Qwen2.5-VL-7B 和 Qwen3-VL-8B，不再使用 3B。

```bash
conda activate guvp-vlm
python vlm_pairwise.py --images-dir /path/to/street_images --max-pairs 20 \
  --mec-samples 3 --temperature 1.0 \
  --audit-output output/vlm_smoke/pair_audit.csv \
  --output data/vlm_pairwise_votes.csv
python main.py --input data/vlm_pairwise_votes.csv --output-dir output/vlm --dimension visual_pollution
```

图片文件名会作为 `image_a` / `image_b` ID。若需要固定配对，请提供包含 `image_a,image_b` 文件名列的 CSV，并加上 `--pairs path/to/pairs.csv`。断点续跑以 audit 文件为准，并检查模型、提示词版本及 SHA-256、标签布局、图片分辨率、量化和 offload 设置，防止不同实验误接到同一输出。

### MEC+BPC V6 正式运行

7B 与 Qwen3 8B 必须分别启动、分别保存、分别计算排名：

```bash
./scripts/run_7b_formal_and_score.sh
./scripts/run_qwen3_vl_8b_formal_and_score.sh
```

当前优先运行 7B，Qwen3 8B 暂缓。7B 脚本会先校验图片数、配对数、重复配对、每图参与次数和提示词哈希，并用文件锁及 GPU 进程检查阻止重复启动。运行时可在另一终端查看进度：

```bash
./scripts/monitor_7b_formal_abba.sh
```

固定配对表 `data/vlm_pairwise_formal_balanced24_v6_3000.csv` 包含 3,000 个无序图片对；250 张图每张恰好参与 24 对，满足每图 23–25 次的要求。MEC k=3 与 BPC 组合后，每个模型独立执行 18,000 次证据优先方向采样。

7B 输出位于 `data/vlm_pairwise_7b_formal_mec3_bpc_v6_3000.csv` 和 `output/vlm_7b_formal_mec3_bpc_v6_3000/`；Qwen3 8B 输出位于 `data/vlm_pairwise_qwen3_vl_8b_formal_mec3_bpc_v6_3000.csv` 和 `output/vlm_qwen3_vl_8b_formal_mec3_bpc_v6_3000/`。每个目录中的 `pair_audit.csv` 保留所有校准、平局及无效响应，`pair_audit_summary.json` 与 `PAIR_AUDIT_REPORT.md` 汇总 BPDE、校准分差与残余位置分数优势；两个模型的票不会在正式评分阶段合并。

## 3B/7B正式运行对比

2026-08-23完成的Qwen2.5-VL-7B正式运行及其与3B归档运行的跨模型审计见
[`DGX_SPARK_MIGRATION.md`](DGX_SPARK_MIGRATION.md#5-qwen25-vl-7b-formal-run-and-cross-model-audit)。
可复现摘要、逐图/逐对比较和图表保存在`output/vlm_3b_vs_7b/`。该审计发现明显的
A/B位置偏差，因此在完成交换位置复核前，两次运行都不应直接视为VP ground truth。

### 本地 Qwen3-VL 8B 4-bit 比较

RTX 5060 8GB 使用官方 `Qwen/Qwen3-VL-8B-Instruct` 的 bitsandbytes 4-bit NF4
二次量化版本，并把输入嵌入层和输出头卸载到 CPU。首次运行会从 Hugging Face
下载模型；单对冒烟测试：

```bash
./scripts/run_qwen3_vl_8b_smoke.sh
```

确认冒烟测试成功后，可用相同的正式配对表独立运行并自动计算 Qwen3 8B 分数：

```bash
./scripts/run_qwen3_vl_8b_formal_and_score.sh
```

正式脚本支持断点续跑；已有输出时会自动添加 `--resume`。

运行中可在另一个终端每 10 秒刷新进度、速度、ETA 和显存状态：

```bash
./scripts/monitor_qwen3_vl_8b.sh
```

Qwen3-VL 8B 正式运行已完成全部 15,562 个配对。结果保存在
`data/vlm_pairwise_qwen3_vl_8b_formal_v2.csv` 和
`output/vlm_qwen3_vl_8b_formal_v2/`；与 Qwen2.5-VL 3B/7B 的对比位于
`output/vlm_qwen2_5_3b_vs_qwen3_8b/` 和
`output/vlm_qwen2_5_7b_vs_qwen3_8b/`。

完整审计见 [`QWEN3_RESULTS_AUDIT.md`](QWEN3_RESULTS_AUDIT.md)。审计确认运行文件
完整，但发现三种模型均存在显著 A/B 位置偏差；在完成交换位置复核前，这些排名
不应直接视为视觉污染 ground truth。

三模型的双向交换、同图零信息和视觉访问对照实验见
[`AB_POSITION_EXPERIMENT.md`](AB_POSITION_EXPERIMENT.md)。实验进一步确认三个模型
都能感知图片，但原来的标签后置布局会使 3B 和 Qwen3 无法稳定对应 A/B；改为
标签前置后，7B 和 Qwen3 的自然图交换一致率分别达到 86.9% 和 80.6%，3B 仍受
强位置先验支配。

在极短提示词中加入少量可见 VP object 例子的后续实验见
[`OBJECT_CUE_PROMPT_EXPERIMENT.md`](OBJECT_CUE_PROMPT_EXPERIMENT.md)。该提示在 7B 上保持
A/B 平衡并将换位一致率提高到 88.1%；Qwen3 8B 的换位一致率为 76.2%。后续实验仅使用
7B 和 Qwen3 8B，并要求每个图片对通过 AB/BA 双向复核后才计票，3B 不再参与。
