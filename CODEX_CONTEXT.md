# GUVP ViT→VP 会话交接

最后更新：2026-10-25。

## 现在要记住什么

用户希望用 **DINOv3 ViT-B/16 直接预测视觉污染分数**，替代部署时先语义分割、再 Elastic Net 回归的流程。该方向主要追求预测性能，可以接受缺少语义元素解释。用户已要求设计实验，并将其保存为未来 Codex 可读取的持久上下文。

详细方案唯一入口：[VIT_VP_EXPERIMENT_PLAN.md](VIT_VP_EXPERIMENT_PLAN.md)。本文件只记录摘要与进度。启动说明见 [AGENTS.md](AGENTS.md)。

用户于 2026-10-20 提出扩展研究构想：在人类校准标签基础上，检验可复现的新 taxonomy、DINOv3 跨城市/国家预测、街区热点及独立社会指标关联。总体方法计划为 [PLAN_20261020.md](PLAN_20261020.md)，它是未来研究提案，不是已执行实验的记录；不自动授权新的全球数据采集、付费服务、招募评分者或训练任务。该扩展与现有 1,000 图 ViT 研究的边界已补充到 `VIT_VP_EXPERIMENT_PLAN.md`。

2026-10-20：按用户要求保存 V9 概念提示词草案：[prompts/vp_pairwise_definition_v9_draft.txt](prompts/vp_pairwise_definition_v9_draft.txt)。它参考 Scientific Reports 2026 论文中的七类可见污染线索，同时保留整体感知比较任务，并明确分数锚点、普通城市元素不自动算污染、只依据可见证据。**这是未测试草案**；不覆盖 V8，不改变已有 V6/V8 标签来源，也未用于已完成或运行中的推理。

2026-10-20：按用户进一步明确的范围启动 1,200 图 V9 标签实验，仅进行 Qwen VLM + V9 提示词 + 成对比较 + TrueSkill，不运行 ViT。详细协议见 `VIT_VP_EXPERIMENT_PLAN.md` 的 11.2 节；V9 是独立候选标签版本，未取代原 V6/V8 标签。

## 已核实的数据和既有成果（2026-10-19）

- 项目路径：`/home/colin/GUVP_pairwise_ranking`。
- 元数据：`data/expansion_750/images_manifest_1000.csv`。
- 图片：`data/expansion_750/street_images_1000/<uuid>.jpg`。
- 标签：`output/vlm_expansion_1000/perception_scores.csv`；目标列 `visual_pollution_score_0_10`，越大越污染。
- 1,000 图、20 个欧洲城市、每城 50 图；UUID 与分数一对一匹配，图片已逐一解码并完成内容哈希与近重复审核。
- 来源为 Mapillary 912 图、KartaView 88 图；80 张标记为全景图。
- 本地标签审计确认教师为 Qwen2.5-VL 7B，提示词 **v6**。不要将 README 中 HPC 的 V8 工作流当成本地现有标签的版本证据。
- 比较审计：`output/vlm_expansion_1000/pair_audit.csv`；有效投票：`data/expansion_750/vlm_pairwise_votes.csv`。
- 6,500 对中有 5,146 个有效非平局、270 个平局、1,084 个无效对；每图有效比较 6–13 次。
- 投票 winner 表示更干净；TrueSkill mu 越大越干净，当前 VP 分数是对 mu 做反向 min-max 缩放。标量目标不能弄反方向。
- 旧语义特征：`output/segmentation_expansion_1000/elastic_net_ready.csv`，来自 Mapillary Vistas v2 的 DeepLabV3+ R50 136k 微调权重。
- Mapillary Vistas v2 原始训练/验证数据已归档至外置盘 `/media/colin/白/GUVP_archive/mapillary_vistas`；原路径 `/home/colin/datasets/mapillary_vistas` 是指向该位置的符号链接。仓库旧分割配置仍能沿用原路径，但外置盘未挂载时分割训练会无法读取原始集。当前 ViT→VP 实验仅使用已导出的 124 类语义 CSV 作基线，不读取 Mapillary 原始图像。
- 历史 Elastic Net 外层折均值：MAE 1.5503、RMSE 1.9078、R² 0.3095、Spearman 0.5610；来源 `output/vp_pollution_elastic_net_expansion_1000/`。这是已见城市设置的历史参考，需要共同划分重跑后比较。
- 文件 SHA-256 已写入完整计划；后续实施应重新核对。

## 已设计的核心实验

1. 预训练 DINOv3 ViT-B/16，先冻结编码器：Ridge 是首个模型，小型 MLP 是非线性对照；后续只微调最后 2 个 block。不能跳过冻结基线直接宣称完整微调更优。
2. 主评价是未见城市：5 折外层城市互斥，每折约 800 训练/200 测试；训练城市内再做 4 折调参，预设 3 套外层划分。另设已见城市中的空间分组评估。
3. 所有模型使用共同划分；预测头标准化、特征选择、超参数、早停和尺度标定均只能使用训练/内层验证数据。
4. 默认保留完整图像，长边 512、边长适配 patch=16；384 为对照。比较 CLS、patch mean 和拼接，排除 register/storage tokens。具体预算见完整计划。
5. 主目标是预测固定 VLM 标签；这些 TrueSkill 标签由全局比较图共同生成，需披露标签依赖，并增加仅训练子图监督、留出比较边评估的严格验证。
6. 主指标 MAE，辅助 RMSE/R²/Spearman/Pearson、城市与来源等分组误差；用按城市的配对 bootstrap 比较同一批 OOF 预测。模型家族选择也放在内层。
7. 当前 0–10 是样本内相对尺度；部署固定标签版本和缩放，不能对每批新图重新缩放。对人类感知的结论需要独立人工验证。
8. 语义特征＋Elastic Net 是重跑对照。Transformer 架构优势如要单独主张，需要额外控制预训练等因素的架构比较。

## 当前实施状态

| 阶段 | 状态 | 已有证据 / 未完成工作 |
| --- | --- | --- |
| 实验设计与持久化 | 已完成 | 完整计划、项目入口、本交接文件；`/home/colin/AGENTS.md` 提供从用户主目录启动的阅读入口 |
| P0 数据冻结与审核 | 已完成 | 1,000 张解码成功；唯一哈希无重复；人工审核 4 个 dHash 候选，确认 3 对近重复、1 对误报；产物在 `output/vit_vp_1000_v1/` |
| P1 固定划分 | 主方案已完成 | 三个种子各 5 折未见城市外层和 4 折内层；空间分组评估仍待建立 |
| P2 冻结表征基线 | 已完成 | DINOv3 ViT-B/16 严格加载；384/512 特征缓存；嵌套 Ridge 与语义 Elastic Net 共同城市折比较已完成 |
| P3 非线性与有限微调 | 已完成 | 冻结 MLP、最后两层微调、3,000 条 OOF 配对核验及 2,000 次按城市配对 bootstrap 均完成；报告已更新 |
| P4 稳健性与严格隔离 | 核心评估已完成 | 250m 同城空间分组、100m 缓冲敏感性、训练比较边隔离均已完成；来源/全景细分、标签顺序敏感性和学习曲线仍待补充。另启动 4 个未见欧洲城市的外部一致性评估，Qwen V6 评分在运行中 |
| P5 选型与部署 | 已完成微调模型全数据拟合及单图推理 | `deployment/finetuned_last2/` 含 3 个种子权重和配置；`vit_vp/predict_finetuned.py` 支持单图/目录并校验哈希；GPU 单图 smoke inference 通过 |

### 当前结果、资源与下一步

- 冻结 Ridge 的未见城市 OOF：MAE 1.2447、RMSE 1.5679、R² 0.5340、Spearman 0.7310（3 次重复的均值）。同折语义 Elastic Net：MAE 1.6155、RMSE 1.9915、R² 0.2482、Spearman 0.5351。结果见 `output/vit_vp_1000_v1/REPORT.md`；标签来自 VLM 比较图，不是人工真值。
- 冻结 Ridge 目前选定的全数据部署输入为 384 长边 patch mean，alpha=1000；部署产物是初步可运行版本，不应把全数据拟合结果当泛化性能。
- 冻结特征 MLP 三重复平均：MAE 1.2522、RMSE 1.5801、R² 0.5267、Spearman 0.7240。它在每个外层折的内层选择中都选择 patch mean；对 Ridge 的逐图平均 OOF 配对城市 bootstrap（5,000 次）差值 MLP−Ridge：MAE +0.0003（95% CI −0.0211 至 +0.0226），RMSE +0.0061（−0.0182 至 +0.0302），Spearman −0.0057（−0.0180 至 +0.0051）。没有可辨别的优势；部署暂留 Ridge。MLP 代码与结果在 `vit_vp/fit_frozen_mlp.py` 和 `output/vit_vp_1000_v1/frozen_mlp/`。
- 同城新地点空间嵌套评估（250m city-centered grid，3 repeats）：Ridge MAE 1.217 ± 0.009、RMSE 1.541 ± 0.007、R² 0.550 ± 0.004、Spearman 0.741 ± 0.005；语义 Elastic Net MAE 1.601 ± 0.017。100m 训练隔离缓冲敏感性结果几乎相同（Ridge MAE 1.217 ± 0.010；每折训练保留 753–779 图）。这一评估仍预测全局冻结的 Qwen/TrueSkill 标签。
- 严格比较边隔离实验：每个训练折只用两端都在训练子集的有效比较重算 TrueSkill 训练标签；内层用验证子集内部比较边选型；测试只评价两端都在外层留出城市内的有效比较。每次重复约 1,025 条测试边；DINOv3 Ridge pairwise accuracy 0.759 ± 0.013，语义 Elastic Net 0.685 ± 0.012。外层训练比较图覆盖全部 800 张并连通；测试边图 4–10 个分量，12–28 张测试图片在每折没有测试比较边，因此结论仅针对观测到的留出边。产物在 `output/vit_vp_1000_v1/edge_isolated_unseen_city/`。
- Mapillary 原始训练/验证集（40,001 个文件，约 19 GB）已完整复制至 NTFS 数据盘 `/media/colin/白/GUVP_archive/mapillary_vistas`；逐文件 checksum 比对无差异后移除系统盘副本，并创建原路径兼容软链接。系统盘从约 3.5 GB 可用升至约 22 GB；外置盘还剩约 30 GB。外置盘需保持挂载以继续使用原始集。
- 最后两层微调使用同一未见城市嵌套划分：3 次重复 × 5 个外层城市折，内层 4 折选择 384/512 分辨率及 3 个 backbone 学习率，最长 60 epoch、patience 20；冻结前 10 个 block 的 prefix 特征已缓存。逐折结果在 `output/vit_vp_1000_v1/finetune_last2/`，配置记录于 `resume_config.json`。2026-10-19 15:59 左右完成训练；360/360 个内层候选、15/15 个外层折和 3,000 条 OOF 预测齐全，`run_metadata.json` 记录 CUDA 运行及峰值分配显存 2,844,343,808 bytes（约 2.65 GiB）。三重复微调均值：MAE 1.1857（SD 0.0100）、RMSE 1.4990（SD 0.0075）、R² 0.5741（SD 0.0043）、Spearman 0.7531（SD 0.0031）、Pearson 0.7578（SD 0.0028）。
- 2026-10-19 16:53 完成配对城市聚类 bootstrap：`vit_vp/compare_finetune_ridge.py` 验证了 3,000 条完全匹配的 repeat/image OOF 行、15 个完整测试折、每折 4 个城市/200 图、每城每重复 50 图；bootstrap 2,000 次、20 个城市为 cluster、seed 20261019。微调相对同折冻结 Ridge 的三重复平均 OOF：MAE 1.1768 vs 1.2312，差 −0.0543，95% CI [−0.0794, −0.0271]；RMSE 差 −0.0622 [−0.0903, −0.0330]；Spearman 差 +0.0169 [+0.0052, +0.0274]；相对 MAE 降低 4.41% [2.22%, 6.40%]。相对同折语义 pooled Elastic Net 的 MAE 差 −0.4282 [−0.5359, −0.3383]，降低 26.68% [21.81%, 31.90%]，超过计划中的 5% 实用参考。城市 bootstrap 对 20 个 cluster 的不确定性是近似估计；重复划分不是独立样本。标签仍为全局 Qwen/TrueSkill VP，不是人工评分。报告 `output/vit_vp_1000_v1/REPORT.md` 已更新；配对 OOF、指标、bootstrap 表和元数据均在 `finetune_last2/`。
- 2026-10-19 17:00 前完成全数据部署拟合：从 15 个 outer-fold 内层选择记录汇总得到 edge 512（11/15 众数）、backbone LR `1e-5`（15/15）、head LR `1e-4`、weight decay `0.01`，epoch 使用 `refit_epochs` 中位数 16；在全部 1,000 图上以 seeds 42/137/2026 各训练一个模型。模型产物 `output/vit_vp_1000_v1/deployment/finetuned_last2/{model.json,seed_*.safetensors}`，三份权重各约 57.1 MB，元数据记录 SHA-256、标签均值/标准差和训练环境。新增 `vit_vp/train_final_finetune.py` 与 `vit_vp/predict_finetuned.py`，后者对三 seed 分数取算术平均，并校验基础 ViT 与微调权重哈希。`py_compile` 通过；CUDA 单图推理通过，示例训练图输出 5.6607（仅证明模型装载/推理成功，不是泛化性能）。全数据模型训练拟合结果不可作为泛化估计，正式性能仍引用嵌套 OOF。

上表的“未开始”指本方案的实施记录。若后续会话发现其他地方已有相关产物，应核对其配置、标签和来源后更新，不能仅凭此表断言磁盘上不存在任何 ViT 权重。

## 当前外部城市一致性评估（2026-10-20，已完成）

- 用户要求从 4 个新增城市抽样 200 张图，分别用 Qwen + V6 流程和已部署 ViT 预测，再对独立归一化分数做相关性分析。该实验是外部标签一致性/排序关联分析，不训练或调整 ViT。
- 外部图像来自 KartaView（四城各 50 张），因为当前未配置 Mapillary token 且候选池筛选后 KartaView 提供了满足数量的合格序列。城市为 Chisinau、Stuttgart、Bern、Hildesheim。该来源与原训练样本（以 Mapillary 为主）不同，城市与来源效应无法在本实验完全拆开，因此结果要作为来源变化下的外部一致性证据谨慎解读。
- 抽样依据：good/day/no glare/reflection、每个序列至多一图、空间网格轮转后以固定随机种子 20261019 抽样。图片下载后均可解码且 SHA-256 唯一；相对于训练集 1,000 张图的 dHash 审核没有近重复候选。清单、抽样元数据、候选和下载日志在 `data/external_vp_4city_200/`。
- 对 200 张图构造固定连通图：每图恰有 13 次参与、总 1,300 个无序对，其中同城 600、跨城 700；种子 20261020。配对清单为 `data/external_vp_4city_200/pairs_200_13_per_image.csv`。
- ViT 推理已完成：调用三种子 DINOv3 ViT-B/16 last-two-block 微调集成，不再训练，预测文件 `output/external_vp_4city_200/vit_predictions.csv`。
- Qwen V6 评分已完成：参数为 Qwen2.5-VL-7B-Instruct、FairEval MEC+BPC V6、MEC=3、batch=1、复用 prefix cache、max_pixels=401408、temperature=1、max_new_tokens=192、seed=42、NF4 double quantization + BF16、关闭 CPU offload；设备 RTX 5060。1,300/1,300 对、7,800 个 MEC 方向样本（2,600 个生成批次）完成；得到 1,014 条校准票，286 对为无效/平局/排除；TrueSkill 评分覆盖 200 张图。运行耗时约 18,826 秒，峰值 CUDA allocated 5.88 GiB。产物在 `output/vlm_external_4city_200_v6/`。
- 外部关联分析已完成：运行 `vit_vp/analyze_external_correlation.py`，Qwen 和 ViT 分数按 image_id 一对一覆盖 200/200 图，并分别在该 cohort 上 min-max 到 [0,10]。总体 Pearson r=0.7646、Spearman ρ=0.7346；按城市分别的 Spearman：Bern 0.6948、Chisinau 0.7103、Hildesheim 0.7614、Stuttgart 0.7287；先去除城市内均值再合并的 Pearson r=0.7654、Spearman ρ=0.7168。逐图合并结果为 `output/external_vp_4city_200/external_qwen_vit_scores.csv`，指标为 `external_correlations.csv`，分析说明为 `external_analysis_metadata.json`。独立 min-max 不改变 Pearson/Spearman；MAE/RMSE 依赖当前归一化尺度，城市中心化行则表示中心化分数差异，均不是外部绝对校准误差。
- 本实验比较的是 ViT 与 Qwen V6 生成标签的一致性，不等于与人类感知或人工标注一致。样本只有 4 城、图片来自 KartaView，而训练数据主要来自 Mapillary，城市效应与来源效应不能完全拆开；相关数值作描述性证据，不据此宣称全球泛化或使用逐图独立假设的显著性。

## 当前 V9 1,200 图 VLM 成对评分（2026-10-20，已完成）

- 用户明确的范围：Qwen VLM + V9 提示词 + pairwise comparison + TrueSkill 生成 VP score；不运行 ViT 推理、微调或相关性分析。
- 输入为 1,000 张主数据（20 城）+ 4 个外部城市各 50 张，共 1,200 张唯一图片；来源组成 Mapillary 912、KartaView 288。输入清单分别为 `data/expansion_750/images_manifest_1000.csv` 与 `data/external_vp_4city_200/images_manifest.csv`。
- `vit_vp/prepare_v9_1200.py` 已创建 `data/vlm_v9_1200/`，图片用符号链接暂存，不复制约 325 MB 外部图像；已对所有 1,200 张执行图片解码校验。固定随机重标号的连通 13-regular 比较图：7,800 对，每图恰好 13 对，种子 20261020；其中 339 对同城、7,461 对跨城。配对图已覆盖 24 城，每城 50 张。
- 正式 MEC+BPC preflight 已通过：7,800 对、46,800 个方向样本。当前运行实际使用优化版提示词 `prompts/vp_pairwise_definition_v9_optimized_draft_20261020.txt`，版本 `vp-definition-score-v9-optimized-draft-20261020`，strip 后 SHA-256 `b9ef4a76b1d4b46c056793bb706dac0d783e23dc4be14db3504a4c7e131b26cd`。注意 V9 输出分数越低代表越干净，故运行参数使用 `lower_is_cleaner`。
- 推理配置沿用 200 图成功设置：本地 RTX 5060、Qwen2.5-VL-7B-Instruct、本地 HF 权重缓存、batch=1、prefix cache、max_pixels=401408、MEC=3、temperature=1、max_new_tokens=192、seed=42、NF4 double quantization + BF16（支持时）。计算时间规划按 200 图实测外推约 31 小时；这是粗略估计，需用新任务进度重估。
- 暂存和固定配对文件：`data/vlm_v9_1200/images_manifest_1200.csv`、`pairs_1200_13_per_image.csv`、`pair_design.json`。正式运行脚本：`scripts/run_vlm_v9_1200_and_score.sh`；实际结果目录：`output/vlm_v9_1200_gpuonly_optimized/`。
- V9 任务于 2026-10-20 16:14 完成：审计覆盖 7,800/7,800 对（46,800 个方向样本），其中 6,639 张有效校准票，1,161 对为平局/无效；推理耗时约 44,136 秒（12.26 小时），峰值 CUDA allocated 5.94 GiB。TrueSkill 产物覆盖 1,200/1,200 张图，`visual_pollution_score_0_10` 按污染方向 min-max 至 [0,10]。Q-score 与 TrueSkill 汇总相关为 Pearson r=0.9654、Spearman ρ=0.9646；这反映同一投票数据的两种聚合方式，不是独立验证。审计、票数、进度、耗时、排序摘要及分数分布图均在上述结果目录。
- V6/V9 逐图对照已完成：V6 主图 1,000 张与外部 200 张均和 V9 1,200 张 ID 全匹配。因为 V6 两批来自独立比较图，分别报告 cohort 内排序：主图 Spearman ρ=0.687，外部图 ρ=0.580；按各自 cohort 将 V9 重缩放至 0–10 后 MAE 分别为 1.338、1.412。两个 cohort 的 top 10% 高污染图片重合分别为 51/100、10/20；24 城城市 Spearman 中位数 0.632（范围 0.354 Tallinn–0.795 Krakow）。逐图合并表和完整限制见 `output/vlms_v6_v9_score_comparison_1200/REPORT.md`。
- V9 仍是未做人工校准的提示词草案，生成分数只是这批图和该比较图内的模型派生相对标签；不得称为人类 ground truth，也不覆盖 V6/V8。

## 当前 V9 ViT 图像级留出评估（2026-10-20，已完成）

- 用户要求沿用 DINOv3 ViT-B/16 last-two-block + patch-mean MLP 方案，清除之前的微调权重，以 V9 标签前 1,000 张训练、最后 200 张作图像级留出评估。实际从固定原始 DINOv3 预训练 checkpoint 重置训练；没有加载 V6 微调权重。超参数固定为 edge 512、backbone/head LR 1e-5/1e-4、weight decay .01、16 epochs、seeds 42/137/2026；前 1,000 张上做 3 repeats × 5 城市外层折，随后以全部 1,000 张拟合三种子集成。
- 200 张留出图没有进入训练脚本、内部划分、选型或早停。数据冻结和图结构审计由 `vit_vp/prepare_v9_holdout_1000_200.py` 生成，训练由 `vit_vp/train_v9_1000_holdout.py` 完成，单独留出预测由 `vit_vp/evaluate_v9_200_holdout.py` 完成；监控脚本为 `vit_vp/monitor_v9_training.sh`。
- 内部 3-repeat OOF 指标：MAE 1.1151/1.1137/1.1166，Spearman ρ 0.6665/0.6615/0.6655，Pearson r 0.6948/0.6908/0.6926（各 repeat 均在 1,000 张开发图上做城市留出）。
- 200 张图像级留出：MAE 1.0531、RMSE 1.3261、R² 0.3343、Spearman ρ 0.5482、Pearson r 0.5919、平均预测偏差 +0.1043；四城 macro Spearman ρ 0.5329。城市明细、逐图预测和完整解释在 `output/vit_vp_v9_holdout_1000_200/REPORT.md`、`holdout_metrics_by_city.csv`、`holdout_predictions.csv`。
- V9 标签来自同一张 1,200 图连通比较图，训练/留出分区间存在 1,790 条有效比较边；留出图内部有效比较图有 60 个连通分量。故本实验严格隔离了最后 200 张图与 ViT 拟合和模型选择，结论称为“图像级留出评估”；标签来源仍共享比较图信息，不能称为独立标签验证或人工感知验证。该限制已写入 `partition_audit.json` 和报告。

## 当前 V9 ViT 城市隔离 900/300 评估（2026-10-20，已完成）

- 用户要求重设计为 900 训练 / 300 测试，并重新从原始权重微调。为防止同城泄漏，测试集按完整城市留出：保留原先四个外部城市 Bern、Chisinau、Hildesheim、Stuttgart（200 张），再按固定 seed 20261020 从原 20 城抽出 Madrid、Prague（100 张）；其余 18 个原始城市共 900 张训练。训练和测试城市/图像均无交集。
- 使用与之前相同的 DINOv3 ViT-B/16 last-two-block + patch-mean MLP 架构，从 SHA-256 `9a21ac3d...dc8b` 的原始预训练 checkpoint 初始化；不加载任何 V6/V9 微调权重。配置固定 edge 512、LR 1e-5/1e-4、weight decay .01、16 epochs，三种子 42/137/2026；不做测试集选型或早停。准备、训练、评估脚本分别为 `vit_vp/prepare_v9_holdout_900_300.py`、`vit_vp/train_v9_900_300.py`、`vit_vp/evaluate_v9_900_300.py`。
- 300 张总体测试：MAE 1.0684、RMSE 1.3533、R² 0.4032、Spearman ρ=0.6142、Pearson r=0.6464。城市宏平均 Spearman ρ=0.5621。
- 测试子组：四个外部城市 200 张，Spearman ρ=0.5513、Pearson r=0.5929、MAE 1.0562；两个原始队列留出城市 100 张，Spearman ρ=0.6937、Pearson r=0.7129、MAE 1.0926。逐城结果见 `output/vit_vp_v9_holdout_900_300/test_metrics_by_city.csv`。
- V9 目标仍是既有 1,200 图联合比较图的 TrueSkill + 全局 min-max 分数；900/300 间有 2,395 条有效比较边。300 张测试图内部有效比较图有 27 个分量、12 张图没有测试集内部有效边。因此图像和城市对 ViT 完全留出，标签仍受联合比较图构造影响。完整协议和限制见 `output/vit_vp_v9_holdout_900_300/REPORT.md` 与 `partition_audit.json`。
- 用户质疑训练速度后，按其“从头微调”要求在全新目录 `output/vit_vp_v9_holdout_900_300_fresh/` 再跑了一次同协议训练和测试。新 run 从同一 SHA-256 原始 DINOv3 checkpoint 新建模型，没有恢复上次微调文件；仍按既定协议只解冻最后两个 block 和 MLP 头，前缀特征来自原始 checkpoint 缓存。新结果与首次运行相符：300 张总体 Spearman ρ=0.6141、Pearson r=0.6464、MAE=1.0683；外部 200 张 Spearman ρ=0.5511、Pearson r=0.5929。该重复运行确认流程，但不是全 12 个 block 解冻的端到端微调。
- 用户要求 900/300 改用与 1,000/200 相同的 3-repeat×5-fold OOF + final fit 方法。该运行已在 `output/vit_vp_v9_holdout_900_300_cv/` 完成：仅用训练清单中的 900 图/18 城生成内部 city folds，每折 700 或 750 train、200 或 150 validation；每折 3 seeds、16 epochs；15/15 折共 2,700 条 OOF 预测齐全，最终三种子模型也已拟合。内部重复 OOF 均值：MAE 1.1304、RMSE 1.4313、R² 0.4633、Spearman ρ 0.6527、Pearson r 0.6814。最终 300 张六城测试：MAE 1.0683、RMSE 1.3531、R² 0.4033、Spearman ρ 0.6143、Pearson r 0.6465；测试评估于 2026-10-20 完成。训练脚本不读取测试清单；但该测试集此前已用于同配置的 900/300 评估，不能称全新盲测。详细结果在该目录 `REPORT.md`、`test_metrics.json`、`test_metrics_by_city.csv`、`test_predictions.csv`。监测脚本 `vit_vp/monitor_v9_900_300_cv.sh`。
- 后续 LoRA 候选方案已写入 `VIT_VP_EXPERIMENT_PLAN.md` 11.3：在最终两层的 Q/V 投影加 LoRA（r=8、alpha=16、dropout=.05），冻结基础权重并训练回归头；复用相同 3×5 city OOF 与 last-two-block 全量微调对照。方案尚未执行；当前环境未安装 PEFT。现有 300-test 基线结果已公开，未来该集上的 LoRA 数值应作为复用测试集对照，若需要全新确认则需新增封存测试数据。

## 下一步

V9 1,200 图 VLM、1,000/200 图像级留出、900/300 简化拟合及 fresh rerun 均已完成。900/300 的 3×5 OOF 补充运行及最终测试评估也已完成，结果见 `output/vit_vp_v9_holdout_900_300_cv/REPORT.md`。LoRA 候选方案已记录在实验计划 11.3，尚未运行；既有 V6 四城一致性结果保持不变。本周汇总见 `WEEKLY_VIT_PROGRESS_2026-10-20.md`。

代码目录：`vit_vp/`；实验结果目录：`output/vit_vp_1000_v1/`。

## 持续更新格式

每次推进后直接修改上方阶段表，并补充一条简短记录：

`日期｜完成了什么｜配置/数据版本｜产物路径｜实际校验结果｜运行状态或阻塞｜下一步`

维护摘要和完整计划之间的一致性；不要将此文档变成逐条工具调用日志。

### 记录

- 2026-10-19：完成 DINOv3 权重部署、`guvp-vit` 隔离环境、P0 审核、P1 城市划分、冻结 Ridge/Elastic Net/MLP、250m 同城空间评估、100m 缓冲敏感性和严格比较边隔离 OOF。冻结 Ridge 在全部三种验证视角均优于语义基线；MLP未提高 Ridge 表现。Mapillary 原始数据已校验后归档到外置 NTFS 盘，系统盘剩余约22 GB。下一步有限微调最后两个 Transformer block。
- 2026-10-19：启动最后两个 DINOv3 block 的完整嵌套微调；prefix 缓存约 1.97 GB，RTX 5060 显存使用约 3.5 GB。逐内层折原子写入候选分数，配置锁定并可续跑。最先完成的候选内层折分数记录在上方；尚无外层 OOF 或微调对 Ridge 的正式比较，运行中。
- 2026-10-19：完成 last-two-block 微调的 3×5 外层 OOF、配对城市聚类 bootstrap（2,000 次）和报告更新。相对冻结 Ridge 的平均 OOF MAE 降低 4.41%（95% city-bootstrap CI 2.22%–6.40%）；相对语义 Elastic Net 降低 26.68%（21.81%–31.90%）。有限微调评估阶段完成；下一步全数据拟合并制作微调模型推理包。
- 2026-10-19：完成三种子全数据微调部署模型与单图推理脚本；训练 epoch=16 是 15 个 nested-CV refit epochs 的中位数；CUDA smoke inference 成功。最终部署文件见 `output/vit_vp_1000_v1/deployment/finetuned_last2/`，研究报告已同步。下一阶段为来源/全景、标签顺序敏感性和学习曲线。
- 2026-10-20：核验本地无 ViT/Qwen 任务仍在运行、RTX 5060 当前空闲；四城外部 V6 运行已完成，1,300 对结束，输出 200 张 Qwen/TrueSkill 分数；与既有 200 张 ViT 预测合并并完成相关分析。总体 Spearman ρ=0.7346，城市内中心化后 ρ=0.7168。将分析表的误差列名改为适用于普通 min-max 和中心化尺度的通用名称后重写产物。结果仅说明该四城 KartaView 子集上 ViT 与 Qwen 标签的一致程度；完整限制及产物见“当前外部城市一致性评估”。
- 2026-10-20：准备并通过正式 preflight 的 V9 1,200 图 MEC+BPC 任务。固定 7,800 对、每图 13 对、图连通、MEC=3、AB/BA、V9 prompt hash 已记录；启动 Qwen2.5-VL-7B-Instruct 推理及 TrueSkill 后处理。单独产物目录为 `output/vlm_v9_1200/`；当前运行状态需看其 progress.log/run.log，而不把预估工期当成完成承诺。
- 2026-10-20：V9 优化提示词运行已确认停在 337/7,800 对；审计与当前提示词哈希、模型及推理参数全部匹配。03:58 从断点续跑，已通过 preflight，跳过 337 个完整配对并开始后续推理；TrueSkill 后处理待全部配对结束。
- 2026-10-20：V9 优化提示词 1,200 图任务完成，7,800 对全部审计，6,639 有效票，TrueSkill 输出覆盖 1,200 图并按污染方向缩放至 0–10。主产物位于 `output/vlm_v9_1200_gpuonly_optimized/`；分数是本次图内的模型派生标签，不是人类 ground truth。
- 2026-10-20：完成 V6/V9 1,200 图逐图对照，V6 的 1,000 主样本与独立 200 外部样本分层分析；覆盖 1,200/1,200，分别 Spearman ρ=0.687/0.580。结果及逐图 rank shift 产物在 `output/vlms_v6_v9_score_comparison_1200/`。
- 2026-10-20：完成 V9 ViT 1,000/200 图像级留出实验；从原始预训练 DINOv3 权重开始，15/15 内部城市折及三种子全量拟合完成，随后对最后 200 张仅评估一次。留出 Spearman ρ=0.5482、Pearson r=0.5919、MAE=1.0531；报告含 V9 pairwise 图标签跨分区共享信息的限制，产物在 `output/vit_vp_v9_holdout_1000_200/`。
- 2026-10-20：完成 V9 ViT 900/300 城市隔离实验；按整城留出 Madrid、Prague 及四个外部城市，900 张来自其余 18 个原始城市；原始 DINOv3 权重起训，固定 16 epochs、三种子集成。总体测试 Spearman ρ=0.6142、Pearson r=0.6464；外部 200 张子组 Spearman ρ=0.5513、Pearson r=0.5929。结果在 `output/vit_vp_v9_holdout_900_300/`。
- 2026-10-20：按用户要求以新目录从原始 DINOv3 checkpoint 完整重跑相同 900/300 last-two-block 微调，不续用既有微调权重；指标与首次运行基本一致。结果在 `output/vit_vp_v9_holdout_900_300_fresh/`。若后续要求全 12 block 解冻，应视为不同训练协议。
- 2026-10-20：用户要求将 LoRA 设计保存为后续方案。已在 `VIT_VP_EXPERIMENT_PLAN.md` 11.3 记录 LoRA-final-two Q/V 配置、900 张训练集 3×5 city OOF 对照、测试集重用限制和未安装 PEFT 状态；不影响当前 full last-two-block CV。
- 2026-10-20：900/300 与 1,000/200 相同方法的补充 CV 完成：900 张开发图 3×5 city OOF 共 2,700 行，最终全 900 三种子模型已拟合；对 300 张六城留出集评估一次。OOF 三重复均值 MAE 1.1304、Spearman ρ 0.6527；测试 MAE 1.0683、Spearman ρ 0.6143。详情 `output/vit_vp_v9_holdout_900_300_cv/REPORT.md`。测试集曾用于前一轮固定配置结果，后续不能当作全新盲测。
- 2026-10-20：生成并更新本周 ViT→VP 进展报告 `WEEKLY_VIT_PROGRESS_2026-10-20.md`，汇总模型设计与部署、V6/V9 训练评估、外部一致性和 900/300 OOF/测试最终结果；LoRA 明确标为尚未执行。

## 学校 HPC Q10 任务

当前 HPC 任务焦点：**只运行 Qwen2.5-VL-7B-Instruct + V8 提示词 + 1,000 张图片/6,500 对**；用户明确暂不做其他模型项目。HPC 项目路径为 `/share/home/sgxcw3/GUVP_pairwise_ranking`，操作方式是直接修改该现有目录，不克隆 GitHub。

### 已核实的 Q10 环境与数据

- 用户可使用的分区只有 `Q10`。探测结果（2026-10-19）：Slurm 23.11.10；节点 `node31`；`gpu:l20:8`；64 CPU、512000 MB RAM；分区最长 21 天；账户关联 `sgxcw3`。当时节点 8 张 GPU 均已分配。
- HPC 私有配置 `jobs/q10.env`：`Q10_PARTITION=Q10`、`Q10_GRES=gpu:l20:1`、8 CPU、32G RAM、`Q10_TIME=13:00:00`、`Q10_ACCOUNT=sgxcw3`、`HPC_MODULE=none`、`CONDA_ENV_NAME=guvp-vlm`，Python 路径 `/share/home/sgxcw3/miniforge3/envs/guvp-vlm/bin/python`。
- 已核实环境：Python 3.11.17、PyTorch 2.10.0+cu128、Transformers 5.20.1、bitsandbytes 0.50.2、Accelerate 1.15.0、Pillow 13.0.0。登录节点 `torch.cuda.is_available()` 为 False 是预期现象；需等计算节点分配 GPU 后验证实际 CUDA。
- `Qwen/Qwen2.5-VL-7B-Instruct` 权重已在 HPC HF 缓存中，snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5`；本次提交前以 `snapshot_download(..., local_files_only=True)` 验证成功。
- HPC 数据检查通过：图片 1,000 张可解析，manifest 1,000 行，唯一配对 6,500 对，每图参与 13 对。正式预检通过；提示词版本 `vp-definition-score-v8`，SHA-256 `718e5c4dac829a70e4eb1dab07743a0c7869b3b90ebda6b23b7b464c205acf5b`，方向 `lower_is_cleaner`，总方向采样 39,000 次。
- 正式输出在提交前不存在：`output/vlm_expansion_1000_definition_v8/pair_audit.csv`、`vlm_pairwise_votes.csv`。

### 作业配置、提交与当前状态

- 正式脚本：`jobs/run_qwen25_vl7b_v8_expansion_1000.sbatch`。资源请求为 1 张 `gpu:l20`、8 CPU、32G RAM、13 小时。推理参数：NF4 4-bit + double quantization，支持时 bfloat16，SDPA，`max_pixels=401408`，MEC 每位置 3 次、`--mec-batch-size 3`、temperature 1.0、最多 192 新 token、seed 42；batch 3 时移除了不兼容的 prefix cache。
- 旧任务 **350491** 原在 Q5、通用 `gpu:1`、1 天，处于 `PartitionConfig`；用户于 2026-10-19 执行 `scancel 350491`，随后确认已离开队列。
- 提交前已在 HPC 完成：配置断言、`bash -n`、Python `py_compile`、图片/配对解析与覆盖检查、正式 preflight、权重缓存检查、输出空目录检查；全部通过。`sbatch --test-only` 返回的 353414 只是测试编号，不是正式任务。
- 正式任务 **353415** 已于 2026-10-20 14:31 立即失败（elapsed 0、exit 1）。日志显示它从 `/var/spool/slurm/d/jobs/lib/q10_python.sh` 加载辅助脚本；根因是脚本依据 staged `BASH_SOURCE` 推算项目路径，而 Slurm 实际把脚本放在 `/var/spool/slurm/...`。HPC 项目中的 `jobs/lib/q10_python.sh` 实际存在，且 SHA 与本地相同，并非缺少上传。
- 已修正 V8 1000 sbatch 路径解析，令 `jobs/submit_q10.sh` 导出绝对 `PROJECT_DIR`；直接 `sbatch` 时也采用 `SLURM_SUBMIT_DIR` / 已知 HPC 项目路径。HPC 文件已由用户上传并核对 SHA：`run_qwen25_vl7b_v8_expansion_1000.sbatch`=`a017450feb6aa83ccbb81ad190da95ffdef1a7fb9c4a87425dac000c8d869ae9`，`submit_q10.sh`=`f016abcfa5c5ebf91c6d267748ea9309834b17cfbef5077dc7a7949cf4d96f39`。Bash 语法检查与 Q10 `sbatch --test-only` 均通过。
- 正式作业 **355992** 已于 2026-10-21 02:48:15 至 14:33:43 完成，Slurm `COMPLETED`、ExitCode `0:0`、耗时 11:45:28。日志确认 6,500/6,500 对、39,000 个 MEC 方向样本全部完成；5,504 张有效校准票，996 对平局/无效，1,000 张 perception score 均已输出。产物行数：audit 6,501、votes 5,505、scores 1,001（含表头）。`timing.json` 总 wall 42,302 秒、generate 40,925 秒、峰值 CUDA allocated 6.31 GiB。用户随后提出 GPU-only 要求；该任务脚本当时没有 `--gpu-only`，且默认使用 `device_map="auto"`，因此不得推断模型完全在 GPU 上执行。可在 HPC 用 `awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="cpu_offload")c=i;next} NR>1{n++;if($(c)=="true")t++} END{printf "rows=%d cpu_offload_true=%d\n",n,t}' output/vlm_expansion_1000_definition_v8/pair_audit.csv` 检查审计列，但这只记录配置 flag，不足以证明每层最终放置；旧运行的确切设备映射未保存，无法事后严格核实。

Q10 操作背景见 [`jobs/Q10_SETUP.md`](jobs/Q10_SETUP.md)，但其中早期“尚未探测/未提交”和临时资源参数描述已过期；以本节当前 HPC 实测和任务状态为准。下一步继续检查 353415 最新状态；任务启动后核验 GPU 可见性、模型加载、batch 3 显存和配对进度。不要把 Slurm 预计启动时间当作实际运行或完成时间。

## GPU-only 项目约束（2026-10-22）

- 用户明确要求项目限制为 GPU-only。VLM/ViT 的神经网络训练、推理与生成必须运行在 CUDA 上；CPU offload、CPU fallback 和未经验证的 `device_map=auto` 禁止。CUDA 不可用或模型无法完整放入 GPU 时应立即失败。图片解码、预处理、磁盘 I/O 和 sklearn 等非神经网络统计仍可用 CPU。
- 已将 GPU-only 作为后续项目硬约束写入 `AGENTS.md`，并更新 VLM 共用入口、A/B 审计入口、ViT 模型入口及 Qwen3/V8 HPC runners。Qwen 模型加载后会检查 `hf_device_map`；显存不足时需要申请/选择更大 GPU，不能启用 CPU offload。
- 本机工作区改动尚未同步到学校 HPC。尤其更新过的 `jobs/run_qwen25_vl7b_v8_expansion_1000.sbatch` 需在下一次提交前同步；已完成的作业 355992 无法追溯性改变设备放置。

## 当前 V8 1,200 图 VLM 成对评分（2026-10-22，已完成）

- 用户要求复用 V9 的 1,200 图 comparison + TrueSkill 流程，将提示词替换为现有 `prompts/vp_pairwise_definition_v8.txt`；不运行 ViT。为隔离提示词差异，复用 V9 的 1,200 张图片、固定连通 13-regular 比较图和相同参数。
- 正式运行脚本：`scripts/run_vlm_v8_1200_and_score.sh`；输出：`output/vlm_v8_1200_gpuonly/`。提示版本 `vp-definition-score-v8`，strip 后 SHA-256 `718e5c4dac829a70e4eb1dab07743a0c7869b3b90ebda6b23b7b464c205acf5b`。Qwen2.5-VL-7B-Instruct，MEC=3，batch=1，prefix cache，max_pixels=401408，temperature=1，max_new_tokens=192，seed=42，NF4 double quantization + BF16。
- 用户明确要求禁止 CPU 参与推理。当前 `vlm_pairwise.py --gpu-only` 将 Qwen2.5-VL 的视觉、embedding、语言层、norm/rotary 和 lm_head 显式映射至 CUDA，`cpu_offload=false`；不得添加 CPU offload。当前正式进程已成功加载权重并在 RTX 5060 上生成，显存占用约 7.3 GiB（含桌面进程）。
- 正式比较于 2026-10-21 约 00:20 启动，2026-10-22 03:23 完成。审计覆盖 7,800/7,800 对、46,800 个方向样本；得到 7,196 张校准票，230 对为校准平局、374 对因 AB/BA 解析无效而排除。TrueSkill 分数覆盖全部 1,200 张图片；每图参与比较 8–13 次，中位数 12。真实视觉污染分数 `visual_pollution_score_0_10` 范围 0–10，均值 4.8486、SD 2.2380。
- 运行总耗时 97,501 秒（约 27.08 小时），生成阶段 96,319 秒，峰值 CUDA allocated 5.88 GiB；RTX 5060 全程 GPU-only，`cpu_offload=false`。Q-score 与 TrueSkill 在同一批投票上的聚合相关为 Pearson r=0.9834、Spearman ρ=0.9840；这不是独立验证。主产物在 `output/vlm_v8_1200_gpuonly/`：`pair_audit.csv`、`vlm_pairwise_votes.csv`、`perception_scores.csv`、`timing.json`、`ranking_summary.json` 和分布图。
- 与 V9 共用同一 1,200 张图及 7,800 对图，因此可做配对提示词结果比较；V8 与 V9 分数仍是各自比较图内的模型派生相对标签，不是人工真值。首次 auto-map 启动失败时产生的仅表头文件留在 `output/vlm_v8_1200_gpuonly/startup_attempts/`，不属于正式结果。

## V8 标签上的 ViT 原始权重微调（2026-10-22—24，交叉验证与部署拟合已完成）

- 用户要求沿用既有城市分组评估流程、从原始 DINOv3 模型开始微调。因最新标签是 V8 1,200 图评分，本次固定目标为 `output/vlm_v8_1200_gpuonly/perception_scores.csv` 的 0–10 VP 分数，不加载任何 V6/V9 微调权重。24 城各 50 图，来源 Mapillary 912、KartaView 288；逐图解码及 SHA-256 检查通过，1,200 个图像内容哈希唯一。数据审核、标签来源、固定 split 见 `output/vit_vp_v8_1200_citycv/`。
- 外层：种子 42/137/2026 各做 5 折城市留出；因 24 城不能均分，单折留出 4 或 5 城（200 或 250 张），剩余 19 或 20 城。内层对训练城市做 4 折。划分已固定，保存于 `splits/unseen_city_outer.csv`、`unseen_city_inner.csv`；每张图在每次重复中恰好进入一次 OOF。
- 训练模型：DINOv3 ViT-B/16 LVD-1689M 原始 checkpoint SHA-256 `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`；冻结 blocks 0–9，仅微调 blocks 10–11 和 `768→128→1` patch-mean MLP 头。内层选择 384/512 长边和 backbone LR `{1e-6,3e-6,1e-5}`，最大 60 epoch、patience 20；外层按选中配置和内层最优 epoch 中位数，以 seeds 42/137/2026 重训。
- 前缀缓存以 RTX 5060 CUDA 生成：1,200 图两种分辨率共耗时 111.4 秒，384/512 文件大小分别约 0.80/1.40 GiB。单折 profile 仅用 CUDA，700 张训练、250 张验证，最佳 epoch 45；耗时 105.45 秒，峰值 CUDA allocated 1.48 GiB。按每个候选约 105 秒粗估，360 个内层候选拟合约 10.5 小时，另加 45 个外层三种子重训；正式训练吞吐需按进度更新。
- 嵌套城市 CV 已完成：15/15 外折选型、3,600/3,600 OOF 行。三次重复 OOF 的 Spearman 为 0.8040、0.8060、0.8060（均值 0.8053）；MAE 为 1.0492、1.0435、1.0519（均值 1.0482）。这些是 ViT 预测与 V8 Qwen/TrueSkill 标签的一致性估计，不是人工感知效度。
- 2026-10-24 在 RTX 5060 完成全 1,200 图部署拟合：512 长边（15 折中 13 折选择）、backbone LR `1e-5`、head LR `1e-4`、15 epochs、seeds 42/137/2026。权重和 `model.json` 在 `output/vit_vp_v8_1200_citycv/deployment/finetuned_last2_v8/`；训练拟合不作为泛化性能估计。三种子算术平均为部署预测规则。推理输入复用 `extract_features.preprocess`：EXIF 转正、RGB、保持纵横比、长边 512、尺寸取最近的 16 倍数、Pillow bicubic、ImageNet 均值/标准差，patch mean 排除 CLS 与四个 register tokens。
- V8 全局标签由同一张 1,200 图比较图产生，因此 ViT 城市 CV 隔离未隔离教师标签信息；该限制和模型派生标签属性在任何后续解释中均需保留。

## OECD 欧洲 50 城 Mapillary × V8 ViT 初步分析（2026-10-24，已完成）

- 按用户委托完成 50 个欧洲城市抽样，每城目标 200 张街景，总计 10,000 张；全部 50 城均匹配 OECD 工作簿中的唯一地区行/地区代码。Mapillary 本机 token 配置有效，未暴露 token；图片倾向城市中心 6 km 范围内非全景街道图，每个拍摄序列至多一张，空间网格轮询。不可用、解码损坏和重复内容均做同城替补。最终逐张完整解码通过；Mapillary ID、sequence 和 SHA-256 各 10,000 个唯一。审计与替换记录保存在外置盘 `/media/colin/白/GUVP_archive/oecd_europe_50city_10000/`；manifest 含逐图 Mapillary 图片页链接，若展示/再分发需按 Mapillary CC BY-SA 规则提供可见 attribution。
- 用最新 V8 部署模型三种子（42/137/2026）完成 10,000 图 CUDA 推理，输出每 seed 和等权 ensemble 分数。输入预处理直接复用 `vit_vp/extract_features.preprocess`：EXIF 转正、RGB、长边 512、最近 16 倍数、bicubic、ImageNet normalize，与训练保持一致。主要城市分数为约 1 km 占用网格均值的等权平均；另保留 200 图直接均值敏感性结果。
- 完成 11 个 OECD 主题 Pearson/Spearman 和按国家聚类 5,000 次 bootstrap CI。每主题实际 N=47–50（原表 N/A 按缺失处理）；派生 11 项等权均值只按 46 个完整城市计算，不是官方 OECD composite。总体趋势多为负相关，住房 Pearson r=-0.354/Spearman ρ=-0.290，派生均值 Pearson r=-0.448/Spearman ρ=-0.323；国家聚类区间较宽。全部相关和区间见 `oecd_analysis/correlations.csv`，完整中文报告 `REPORT.md`。
- 解释限制：约 0.805 的 V8 OOF Spearman 是 ViT 对同源 Qwen/TrueSkill 标签的一致性，不是人类感知效度；OECD 城市相关属探索性生态分析，不能证明模型测得真实污染或因果关系。Mapillary 是街景来源，发布/使用需保留其 attribution。
- 工期实测：10,000 张网络下载主任务约 25 分钟，另有不可用/坏图/重复图排查与 67 个替换候选，之后做全量完整解码和哈希审计；RTX 5060 三种子 10,000 图推理约 6 分钟；50 城、11 项、5,000 次国家聚类 bootstrap 分析约数十秒。端到端（含失败排查、替换、复审）约一小时。

记录：2026-10-24｜50 城 10k Mapillary 抽样、最终完整性/OECD 匹配、V8 三种子推理、11 主题相关性分析完成｜V8 全 1,200 图部署模型，512 长边预处理｜外置盘 `oecd_europe_50city_10000/`；模型脚本 `vit_vp/score_oecd_50city_images_v8.py`，相关脚本 `vit_vp/analyze_oecd_50city_vp_correlation.py`｜10k 图片通过完整解码/唯一哈希；50 OECD rows exact｜已完成｜若后续用于正式结论，先补盲法人类评分与抽样敏感性分析。

## OECD 欧洲 50 城扩至 700 张/城样本量敏感性（2026-10-24，已完成）

- 用户要求在原 50 城数据上每城新增 500 张，最终每城 700 张/共 35,000 张。原 10,000 张保留，新增 Mapillary ID/拍摄序列均与原样本不重叠；沿用同一非全景、城市中心半径、空间网格轮询规则。6 km 内候选不足时仅对 Tampere 199 张和 Geneva 147 张扩大到 8/10 km，详见 `supplement_500/combined_sampling_audit.json`。
- 新增下载 25,000 张中 39 张不可用/解码失败；全内容哈希发现 303 张与其他图片重复。均通过同城且序列不重叠的候选替补。最终 35,000 张全量完整解码通过；ID、sequence、内容 hash 均 35,000 个唯一；50 个 OECD 地区代码准确匹配。主图目录、清单与审计：`/media/colin/白/GUVP_archive/oecd_europe_50city_10000/images_manifest_700.csv`、`audit_700.json`、`image_integrity_700.csv`。
- 使用已拟合的 V8 三种子模型对全部 35,000 张 CUDA 推理，`image_vp_predictions_700.csv`。严格复用 V8 512 长边预处理，没有重新训练。
- 700 张/城主分数仍为约 1 km 占用网格均值的等权平均，OECD 分析为 50 城/26 国，按国家聚类 bootstrap。700 张下官方主题 Spearman 较强的为住房 −0.408、环境 −0.339、社区 −0.312；生活满意度 −0.287。原 200 张最强为生活满意度 −0.299、住房 −0.290、社区 −0.262。主题系数有变化，多数配对 country-bootstrap 的 delta CI 仍含 0；11 项等权综合仍是非官方派生分数，使用 46 城 complete cases。
- 城市 score 排名在 200 vs 700 的 Spearman ρ=0.7748；城市分数变化绝对值中位数 0.154、90 分位 0.742，排名位移中位数 5.5 位、平均 7 位、最大 38 位。新增样本提高了空间覆盖（平均占用 1 km cells 从 6.86 到 17.54）。500 次城市内网格/图片层级 bootstrap 得到 700/200 的主题相关 rho 标准差比值均值 0.787（平均降低约 21.3%），各主题约下降 17%–25%。这是对抽样空间不确定性的条件敏感性，不是人类效度或独立 OECD 置信区间。
- 解释：较大样本改善影像抽样引起的城市分数波动，但部分城市和城市排名仍有较大变化；OECD 相关估计随样本改变。图片数提升没有增加 50 城/26 国的独立样本数。不得将这些结果称为对真实视觉污染的验证；V8 OOF ≈0.805 仍是模型对同源 Qwen/TrueSkill 标签的预测一致性。
- 报告：外置盘 `REPORT_200_vs_700.md`；相关产物在 `oecd_analysis_700/` 与 `sample_size_comparison/`。本地运行脚本：`vit_vp/finalize_oecd_50city_700_sample.py`、`vit_vp/compare_oecd_200_vs_700_sample_size.py`、`vit_vp/bootstrap_oecd_city_score_stability.py`；采样和下载脚本支持补样、排除已有序列及设定 supplemental count。

记录：2026-10-24｜50 城增至 700 张/城，补样、全量审核、V8 推理、OECD 相关与 200/700 稳定性 bootstrap 完成｜V8 3 seed，训练一致 512 edge｜`oecd_europe_50city_10000/`｜35,000 张全解码、ID/序列/hash 唯一；城市排序 Spearman 0.775；空间 bootstrap SD 平均下降 21.3%｜完成｜下一步如需强结论，新增独立城市样本或盲法人类标签；仅增加同城图片不能提高城市级独立 N。

## OECD 欧洲扩展至 250 候选城市、700 张/城（2026-10-25，已完成）

- 按用户要求继续在 Linux 对最多 250 个候选欧洲城市做 Mapillary 街景抽样，最终只保留可达到 700 张唯一、完整解码图片的城市。250 个候选中 103 个新增城市在候选预检阶段不足 700 张；下载、解码、内容去重及同城替补后再剔除 12 个无法达标的新增城市。最终保留原 50 城及 85 个新增城市，共 135 城×700=94,500 张，覆盖 132 个唯一 OECD 地区及 28 国。最终所有 UUID、Mapillary 图片 ID、拍摄序列、SHA-256 唯一；全部图像通过解码审计。外置盘清单 `images_manifest_expanded_700.csv`，审计 `audit_expanded_700.json`，完整中文报告 `REPORT_expanded_700.md`。
- 使用最新 V8 全 1,200 图拟合部署模型（DINOv3 ViT-B/16、last-two-block fine-tune、seeds 42/137/2026 等权集成）打分，不重新训练。所有输入遵从部署预处理，长边 512、保持纵横比、最近 16 倍数、bicubic、ImageNet normalize。最终预测 94,500/94,500 与 manifest UUID 精确匹配，均为有限值。预测表为 `image_vp_predictions_expanded_700.csv`。
- 完成 OECD 区域唯一行相关分析：城市先按约 1 km 占用网格等权平均，再将同一 OECD 地区多个城市等权平均；敏感性方法为每城 700 张直接算术平均。按国家聚类 bootstrap 5,000 次。主分析住房 Spearman ρ=-0.435（95% CI [-0.549,-0.312]），环境 ρ=-0.350（[-0.496,-0.190]）；其余主题及 Pearson/Spearman 结果见 `oecd_analysis_expanded_700/correlations.csv`。相关性为探索性地区生态关联，不代表模型对人类感知效度或因果证据；模型 OOF≈0.805 仍指对同源 Qwen/TrueSkill 标签预测的一致性。
- 产物根目录：`/media/colin/白/GUVP_archive/oecd_europe_50city_10000/`。Mapillary 样本由候选可用性筛选，欧洲城市覆盖不均；图片展示/分享需按 Mapillary 适用规则保留 attribution。
- 用户要求可视化三组规模和 11 个 OECD 主题对比，已生成两张图（PNG、SVG）和绘图数据于 `oecd_correlation_visuals/`，复现脚本为 `vit_vp/plot_oecd_correlation_scale_comparison.py`。图一展示 11 主题等权派生均值的 Spearman ρ 与国家聚类 CI；图二逐主题展示三组估计与 CI。扩展实验以 OECD 地区为观测单位，前两项以城市为单位；图中和说明明确这一口径变化，避免将系数变化简单解释为增加每城图片数量的效果。

记录：2026-10-25｜250 候选扩展样本筛选、下载、替补、完整性审计、V8 推理和 OECD 相关分析完成｜135 城×700、V8 3-seed、512 edge｜外置盘 `oecd_europe_50city_10000/`｜94,500 图全部完整且唯一；132 OECD 地区关联完成｜完成｜更强效度结论仍需独立盲法人类标签及更有代表性的城市采样。

## OECD 欧洲扩展至 1,000 张/城（2026-10-25，已完成）

- 按用户要求保留完整 135 城，不因 Mapillary 候选少而剔除城市。在 135×700 基线基础上，每城新增 300 张，最终 135,000 张，135 城全部恰好 1,000 张，映射至 132 个 OECD 地区、28 国。最终 UUID、Mapillary ID、capture sequence、SHA-256 全局唯一，所有文件可完整解码。清单 `images_manifest_expanded_1000.csv`，完整性 `image_integrity_expanded_1000.csv`、`audit_expanded_1000.json`，报告 `REPORT_expanded_1000.md`。
- 新增图优先取 6 km 城市中心范围道路街景；6km、8km、10km、12km 和 Geneva 的 13.5km 样本经补采与内容去重，确保各城均达标。最终新增 40,500 张中，6km 35,596 张、8km 862 张、10km 3,554 张、12km 456 张、13.5km 32 张。取样规则为非全景、质量分可用时≥0.25、每采集序列至多一张、约 1km 网格间随机轮询。半径扩展会纳入部分外围街区，应视为方法异质性。
- 使用 V8 全 1,200 图拟合部署模型，不重训。三个种子（42/137/2026）算术集成；V8 预处理与训练一致：EXIF、RGB、长边512、保持纵横比、最近16像素倍数、bicubic、ImageNet normalize。94,500 张旧分数复用，只对新增40,500张推理；最终预测 135,000/135,000 UUID 精确覆盖，均有限，长边均512。预测文件 `image_vp_predictions_expanded_1000.csv`。
- OECD 关联按约1km占用网格等权→城市等权→同一 OECD 地区多城等权，按国家聚类 bootstrap 5,000 次。主分析住房 Spearman ρ=-0.431（95% CI [-0.553,-0.307]），环境 ρ=-0.369（[-0.509,-0.213]），11主题等权派生均值 ρ=-0.148（[-0.353,0.085]）。135×700 对应派生均值 ρ=-0.157（[-0.338,0.068]）；增加到1,000图/城后估计仅小幅变化，区间高度重叠。见 `oecd_analysis_expanded_1000/` 和 `REPORT_expanded_1000.md`。结果是探索性地区生态相关，不是图像级效度或因果证据。
- 四组样本规模新图为 `oecd_correlation_visuals/01_derived_index_four_scales.png`、`02_oecd_11_themes_four_scales.png`（另有 SVG）；原三组图保留。复现图表脚本 `vit_vp/plot_oecd_correlation_scale_comparison.py` 已更新为四组。最终组装脚本 `vit_vp/finalize_oecd_135city_1000.py`。

记录：2026-10-25｜135 城×1,000 图/城补样、严格去重、最终组装、V8 推理、OECD 相关和四组图表完成｜部署 V8 3-seed，512 edge｜外置盘 `oecd_europe_50city_10000/`｜135,000 唯一有效图，132 OECD 地区｜完成｜更强模型效度结论仍需独立盲法人类标签；135×700 与135×1000的差异不应单独解释为图片数量效应。
