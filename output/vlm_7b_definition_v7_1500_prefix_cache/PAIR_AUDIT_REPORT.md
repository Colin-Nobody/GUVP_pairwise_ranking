# 正式 MEC+BPC 审计摘要

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- 提示词版本：`vp-definition-score-v7`
- MEC：每个位置 k=3，temperature=1.0
- 已完成无序图片对：1500
- MEC 方向采样次数：9000
- 有效校准票：1305（87.0%）
- 校准平局：48
- 无效对：147
- 平均绝对校准分差：1.7531
- 平均 BPDE：0.4168
- 第一位置平均分优势：0.5299

## 状态计数

- `calibrated`：1305
- `calibrated_tie`：48
- `invalid_ab`：68
- `invalid_ba`：74
- `invalid_both`：5
