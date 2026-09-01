# GUVP FairEval MEC+BPC V6

这是 `GUVP_pairwise_ranking` 的独立 FairEval 校准版本，用 Qwen VLM 对街景图像的视觉污染进行两两比较。该目录与仓库根目录的早期单向、AB/BA 和提示词实验相互隔离。

## 方法

本实现适配 Wang et al. 的 *Large Language Models are not Fair Evaluators*：

- **MEC（Multiple Evidence Calibration）**：模型先给出基于可见内容的评价证据，再为两张图片分别给出 1–10 清洁度分数；每个位置独立采样 `k=3` 次。
- **BPC（Balanced Position Calibration）**：每个无序图片对运行 AB 与 BA 两种位置；交换方向的分数会重新映射到实际图片，再对每张图的 `2k` 个分数取平均。
- 校准均分较高的图片表示视觉污染更低，写入 `winner`。校准平局或任一方向没有完整 `k` 个可解析样本时，该对仅保留在审计文件，不进入排名票。
- 输出票继续使用仓库原有 Q-score 和 TrueSkill 聚合，最终 VP 分数越高表示视觉污染越严重。

论文：[ACL Anthology](https://aclanthology.org/2024.acl-long.511/)；官方参考代码：[i-Eval/FairEval](https://github.com/i-Eval/FairEval)。

## 固定配置

完整机器可读配置见 [`MODEL_CONFIG.json`](MODEL_CONFIG.json)。正式 7B 运行使用：

```text
model: Qwen/Qwen2.5-VL-7B-Instruct
quantization: 4-bit NF4, double quantization, BF16 compute
label layout: Image A/Image B labels before their images
max_pixels: 401408
MEC: k=3, temperature=1.0
BPC: AB + BA
max_new_tokens: 192
prompt SHA-256: b34d3f8b72f684ae228edc6ef817b27d07927d8a73b3a5bd172cfc8b2f33fa8e
```

模型权重不在 Git 仓库中；首次运行由 Hugging Face 下载。

## 数据布局

将 250 张街景图片放入：

```text
data/street_images/
```

固定配对清单为 [`data/vlm_pairwise_formal_balanced24_v6_3000.csv`](data/vlm_pairwise_formal_balanced24_v6_3000.csv)：

- 250 张图片；
- 3,000 个唯一无序图片对；
- 每张图片恰好参与 24 对；
- MEC+BPC 共执行 18,000 次方向采样。

街景原图因体积和数据许可不包含在本目录中。

## 安装与测试

```bash
cd faireval_mec_bpc_v6
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-vlm.txt
python -m unittest discover -s tests -v
```

正式运行需要支持 CUDA 的 PyTorch 环境；项目原实验环境为 Python 3.11、RTX 5060 8GB。

## 运行 7B 正式实验

```bash
./scripts/run_7b_formal_and_score.sh
```

脚本会执行输入预检、断点续跑、MEC+BPC 推理、审计汇总以及 Q-score/TrueSkill 排名。已有 `pair_audit.csv` 时会自动恢复；没有时从零运行。

另一个终端监控：

```bash
./scripts/monitor_7b_formal_abba.sh
```

## 运行 Qwen3-VL 8B

先做单对冒烟测试：

```bash
./scripts/run_qwen3_vl_8b_smoke.sh
```

再运行正式实验：

```bash
./scripts/run_qwen3_vl_8b_formal_and_score.sh
```

8GB GPU 配置会对 Qwen3-VL 8B 使用部分 CPU offload。

## 已归档的 7B 正式结果

轻量结果位于 [`results/7b_formal/`](results/7b_formal/)；不包含 13MB 原始模型响应审计。

- 完成 3,000/3,000 对、18,000 次方向采样；
- 有效校准票 2,346（78.2%）；
- 校准平局 138；严格格式无效对 516；
- 最终有效比较覆盖全部 250 张图片，每图 13–23 次，中位数 19；
- Q-score 与 TrueSkill 的 Spearman 相关为 0.9886；
- BPC 后最终 winner 为清单 `image_a` 的比例为 49.7%，但原始方向判断仍有位置偏差，因此不可把单向结果视为 ground truth。

详细限制和审计指标见 [`RESULTS.md`](RESULTS.md)。

## 仓库中有意排除的内容

- Qwen 模型权重和 Hugging Face 缓存；
- 250 张街景原图；
- `pair_audit.csv` 原始生成文本；
- 运行日志、锁文件和临时输出；
- 密钥、token 和本地环境变量。
