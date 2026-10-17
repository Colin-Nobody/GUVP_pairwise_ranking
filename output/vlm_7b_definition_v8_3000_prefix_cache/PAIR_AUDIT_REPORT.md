# 正式 MEC+BPC 审计摘要

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- 提示词版本：`vp-definition-score-v8`
- MEC：每个位置 k=3，temperature=1.0
- 已完成无序图片对：3000
- MEC 方向采样次数：18000
- 有效校准票：2645（88.2%）
- 校准平局：89
- 无效对：266
- 平均绝对校准分差：1.4909
- 平均 BPDE：0.4651
- 第一位置平均分优势：0.4655

## 状态计数

- `calibrated`：2645
- `calibrated_tie`：89
- `invalid_ab`：133
- `invalid_ba`：131
- `invalid_both`：2
