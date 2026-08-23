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

### 本地 Qwen2.5-VL 4-bit 比较

`vlm_pairwise.py` 会以「哪张更干净／视觉污染更低」作为强制二选一题目，输出可直接传给 `main.py` 的 CSV。它默认使用适合 8GB 显存试跑的 `Qwen/Qwen2.5-VL-3B-Instruct`、4-bit NF4 量化和每图最多 512 个视觉 token。

```bash
conda activate guvp-vlm
python vlm_pairwise.py --images-dir /path/to/street_images --max-pairs 20 --output data/vlm_pairwise_votes.csv
python main.py --input data/vlm_pairwise_votes.csv --output-dir output/vlm --dimension visual_pollution
```

图片文件名会作为 `image_a` / `image_b` ID。若需要固定配对，请提供包含 `image_a,image_b` 文件名列的 CSV，并加上 `--pairs path/to/pairs.csv`。

## 3B/7B正式运行对比

2026-08-23完成的Qwen2.5-VL-7B正式运行及其与3B归档运行的跨模型审计见
[`DGX_SPARK_MIGRATION.md`](DGX_SPARK_MIGRATION.md#5-qwen25-vl-7b-formal-run-and-cross-model-audit)。
可复现摘要、逐图/逐对比较和图表保存在`output/vlm_3b_vs_7b/`。该审计发现明显的
A/B位置偏差，因此在完成交换位置复核前，两次运行都不应直接视为VP ground truth。
