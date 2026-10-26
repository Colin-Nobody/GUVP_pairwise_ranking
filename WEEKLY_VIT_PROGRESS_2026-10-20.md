# ViT→VP 周进展报告

**报告周期：**2026 年 10 月 19–20 日（本周截至 10 月 20 日）  
**项目：**DINOv3 ViT-B/16 从街景图像直接预测视觉污染（VP）分数  
**状态：**部署基线、V6/V9 多项评估，以及 V9 900/300 重复城市交叉验证和最终测试均已完成。

## 本周概况

本周完成了 ViT→VP 的模型方案整理、预训练权重上的微调与推理部署，并围绕 V6、V9 两版 VLM 派生标签开展了城市留出和外部一致性评估。模型从单张街景图直接预测 0–10 VP 分数；当前方案不要求人工框选对象或输入对象类别。

已有结果显示，微调后的模型能够预测固定的 Qwen/TrueSkill 标签，并在既有 V6 标签的未见城市交叉验证中优于冻结特征基线和语义 Elastic Net。V9 的 300 张城市留出评估 Spearman ρ 为 0.614。该数字衡量模型对现有 V9 标签的预测表现，不代表与人类判断的一致程度。

## 1. 模型设计与部署

- 主干采用原始预训练 DINOv3 ViT-B/16（LVD-1689M），不是随机初始化。
- 微调时解冻最后两个 Transformer block，前十个 block 保持冻结，并训练 patch-mean MLP 回归头。
- 输入为完整街景图像，模型输出连续 VP 分数。监督目标使用已有 VLM 成对比较经 TrueSkill 汇总并缩放得到的图像级标签。
- 本周用于已部署模型的固定配置为长边 512、主干学习率 `1e-5`、回归头学习率 `1e-4`、weight decay `0.01`、训练 16 轮，使用三个随机种子 42、137、2026 的预测平均。
- 已打包三个种子的微调权重和配置，并提供单图/目录推理入口。部署文件位于 [`deployment/finetuned_last2/`](/home/colin/GUVP_pairwise_ranking/output/vit_vp_1000_v1/deployment/finetuned_last2)，推理脚本为 [`predict_finetuned.py`](/home/colin/GUVP_pairwise_ranking/vit_vp/predict_finetuned.py)。

## 2. 已完成的训练和结果

### 2.1 V6 标签上的主评估与部署模型

1,000 张图像按城市分组进行 3 次重复、每次 5 折的外层评估；每折使用内层城市交叉验证选择设置。每个外层模型只用训练城市拟合，评估时预测未见城市。

| 模型 | MAE | RMSE | R² | Spearman ρ | Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv3 冻结特征 + Ridge | 1.245 | 1.568 | 0.534 | 0.731 | — |
| DINOv3 最后两层微调 + MLP | 1.186 | 1.499 | 0.574 | 0.753 | 0.758 |
| 语义特征 + Elastic Net | 1.615 | 1.992 | 0.248 | 0.535 | — |

以上为三次重复 OOF 结果的均值；微调相对同折冻结 Ridge 的 MAE 降低约 4.4%。模型部署包在完整 1,000 张开发集上训练，供推理使用；泛化性能应引用上表的 OOF 结果，而不是部署包在训练集上的拟合表现。完整报告：[`output/vit_vp_1000_v1/REPORT.md`](/home/colin/GUVP_pairwise_ranking/output/vit_vp_1000_v1/REPORT.md)。

### 2.2 四个外部城市的 V6 标签一致性

已部署 ViT 对 Bern、Chisinau、Hildesheim、Stuttgart 的 200 张图直接预测，并与同批图像的 Qwen V6 成对比较分数对照。总体 Pearson r=0.765、Spearman ρ=0.735；去除城市内均值后合并的 Spearman ρ=0.717。该评估没有重新训练模型。外部图像来自 KartaView，而原训练集主要来自 Mapillary，所以结果同时包含图像来源变化；它反映 ViT 与 Qwen 标签的一致性，不是人工验证。结果文件：[`external_correlations.csv`](/home/colin/GUVP_pairwise_ranking/output/external_vp_4city_200/external_correlations.csv)。

### 2.3 V9 标签与 V6 对照

本周生成了 1,200 张图的 V9 VLM 成对比较标签，并将其与对应图像的 V6 分数对齐。V6 主样本 1,000 张内，两版分数的 Spearman ρ=0.688；外部 200 张内为 0.580。V6 和 V9 使用的比较图及分数尺度不同，因此按队列分别比较，不能将两版分数视为同一绝对标尺。详情：[`V6 vs V9 报告`](/home/colin/GUVP_pairwise_ranking/output/vlms_v6_v9_score_comparison_1200/REPORT.md)。

### 2.4 V9：1,000 张开发图像 / 200 张图像级留出

从原始 DINOv3 预训练权重重新初始化，没有加载 V6 微调权重。前 1,000 张用于训练和内部城市折评估，最后 200 张没有进入 ViT 拟合、选型或早停。200 张留出结果为 MAE=1.053、RMSE=1.326、R²=0.334、Spearman ρ=0.548、Pearson r=0.592。完整报告：[`output/vit_vp_v9_holdout_1000_200/REPORT.md`](/home/colin/GUVP_pairwise_ranking/output/vit_vp_v9_holdout_1000_200/REPORT.md)。

### 2.5 V9：900 张训练 / 300 张城市留出

另一次从原始预训练权重开始的实验，以 18 个城市的 900 张图训练，将 6 个完整城市的 300 张图留出。测试城市与训练城市分开，固定训练 16 轮，并平均三个种子模型。测试结果：MAE=1.068、RMSE=1.353、R²=0.403、Spearman ρ=0.614、Pearson r=0.646。四个外部城市共 200 张的 Spearman ρ=0.551；另两个留出城市共 100 张的 Spearman ρ=0.694。

这 300 张结果曾用于该固定配置的评估，因此后续在同一批图像上做方法比较应标为复用测试集分析，不能再次称为全新盲测。报告：[`output/vit_vp_v9_holdout_900_300_fresh/REPORT.md`](/home/colin/GUVP_pairwise_ranking/output/vit_vp_v9_holdout_900_300_fresh/REPORT.md)。

### 2.6 V9：900 张开发集重复 OOF 与最终 300 张测试

按与前述微调相同的模型配置，对 V9 的 900 张开发图完成 **3 次重复 × 5 折城市留出 OOF**；每折使用三个种子集成。每个开发图在每次重复中恰有一次外层 OOF 预测，共 2,700 条。三个重复的汇总均值和标准差如下：

| 指标 | 三次重复均值 ± 标准差 |
| --- | ---: |
| MAE | 1.130 ± 0.004 |
| RMSE | 1.431 ± 0.004 |
| R² | 0.463 ± 0.003 |
| Spearman ρ | 0.6527 ± 0.0017 |
| Pearson r | 0.6814 ± 0.0020 |

随后在全部 900 张开发图上拟合了三个种子的最终模型，并对六个完全留出的城市共 300 张图评估一次：

| 测试范围 | N | MAE | RMSE | R² | Spearman ρ | Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 总体 | 300 | 1.0683 | 1.3531 | 0.4033 | 0.6143 | 0.6465 |
| 外部四城 | 200 | 1.0561 | 1.3310 | 0.3294 | 0.5512 | 0.5930 |
| Madrid、Prague | 100 | 1.0926 | 1.3964 | 0.5013 | 0.6941 | 0.7129 |

逐城 Spearman ρ：Bern 0.592、Chisinau 0.428、Hildesheim 0.550、Madrid 0.783、Prague 0.496、Stuttgart 0.520。结果表明测试总体排序相关性中等，城市间差异明显；这仍是对固定 V9 标签的预测表现。

测试图像/城市没有进入训练和模型选型，但同一 300 张图已用于先前固定配置的 900/300 结果，因此这次结果属于同一留出集上的复核，不能视作全新盲测。V9 标签仍来自同一张 1,200 图比较图，训练与测试标签之间存在跨组比较信息。完整报告：[`output/vit_vp_v9_holdout_900_300_cv/REPORT.md`](/home/colin/GUVP_pairwise_ranking/output/vit_vp_v9_holdout_900_300_cv/REPORT.md)；逐图预测、逐城指标和 OOF 指标均在同一目录中。

训练脚本：[`train_v9_900_300_cv.py`](/home/colin/GUVP_pairwise_ranking/vit_vp/train_v9_900_300_cv.py)；测试评估脚本：[`evaluate_v9_900_300.py`](/home/colin/GUVP_pairwise_ranking/vit_vp/evaluate_v9_900_300.py)。最终三种子部署权重位于 `output/vit_vp_v9_holdout_900_300_cv/deployment/finetuned_last2_v9/`。

## 4. 下一步方案

- 若要报告新方法相对当前 full fine-tuning 的提升，需在开发集内预先确定比较方案；当前 300 张测试集已被前后两次评估使用，不应再作为全新确认集。
- 对照不同标签版本和城市/来源分组时，保持数据切分和指标定义清楚，避免把共享比较图生成的标签当成独立人工真值。
- LoRA 已作为候选实验设计保存，尚未实施：冻结原始主干，在最后两个 block 的 Q/V 投影上加入 LoRA，并与同折全量微调基线比较。该方案仍用相同图像级 VP 标签，不额外输入对象标签；实施前需核实模块名称并锁定软件依赖。

## 5. 解释范围与限制

1. V6/V9 目标均为 VLM 成对判断经聚合得到的分数，不是人工标注的 ground truth。当前结果说明 ViT 能在一定程度上预测这些固定标签。
2. V9 的 1,200 张标签来自同一张比较图。即使 ViT 训练/测试图像和城市隔离，分区之间仍有跨组比较边；所以这是图像级或城市级留出评估，不是独立生成标签的验证。
3. 外部四城同时改变了城市和图像来源，不能把观察到的差异单独归因于城市迁移。
4. 同一批 300 张测试图已用于前一版 900/300 结果；后续在其上比较改进方法属于复用测试集分析。新的确认性评估需要此前未用于开发的新图像及其标签。
5. LoRA 仅完成设计记录，尚无训练结果；任何“注意力优化”或具体对象识别能力都不能由该设计直接推断。

## 相关文件

- 实验方案与 LoRA 设计：[VIT_VP_EXPERIMENT_PLAN.md](/home/colin/GUVP_pairwise_ranking/VIT_VP_EXPERIMENT_PLAN.md)
- 本周项目交接记录：[CODEX_CONTEXT.md](/home/colin/GUVP_pairwise_ranking/CODEX_CONTEXT.md)
- 已部署推理脚本：[vit_vp/predict_finetuned.py](/home/colin/GUVP_pairwise_ranking/vit_vp/predict_finetuned.py)
- 当前 900/300 OOF 监控脚本：[vit_vp/monitor_v9_900_300_cv.sh](/home/colin/GUVP_pairwise_ranking/vit_vp/monitor_v9_900_300_cv.sh)
