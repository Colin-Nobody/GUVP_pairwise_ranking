# 正式 MEC+BPC 审计摘要

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- 提示词版本：`vp-definition-score-v7`
- MEC：每个位置 k=3，temperature=1.0
- 已完成无序图片对：3000
- MEC 方向采样次数：18000
- 有效校准票：2646（88.2%）
- 校准平局：78
- 无效对：276
- 平均绝对校准分差：1.7300
- 平均 BPDE：0.4229
- 第一位置平均分优势：0.5389

## 状态计数

- `calibrated`：2646
- `calibrated_tie`：78
- `invalid_ab`：137
- `invalid_ba`：133
- `invalid_both`：6
