# GUVP 项目交接记录

> 当前完整状态与 DGX Spark 迁移步骤以 `DGX_SPARK_MIGRATION.md` 为准。
> 本文保留早期实施过程记录；其中“尚未完成”和模型下载状态已经过时。

更新：2026-08-22（Asia/Shanghai）

## 目标与已确认方案

- 目标：使用 Global Streetscapes 的欧洲城市街景，以 Qwen2.5-VL 进行视觉污染（VP）两两比较，随后用本项目的 Q-score 与 TrueSkill 聚合。
- 试验城市已确认：**Amsterdam、Paris、Berlin、Barcelona、Rome**。
- 目标样本：每城约 50 张，合计约 250 张。
- VLM 已选择：`Qwen/Qwen2.5-VL-3B-Instruct`，本地 4-bit NF4 推理，单次一对图片、每图最多 512 个视觉 token。

## 已完成

### 本地 VLM 环境

- Conda 环境：`/home/colin/miniconda3/envs/guvp-vlm`（Python 3.11）。
- 已验证：RTX 5060（8GB）被 PyTorch 正确识别；`torch 2.11.0+cu128`，CUDA 12.8 可用。
- 已安装：`torch`、`torchvision`、`transformers 5.15.1`、`accelerate`、`bitsandbytes 0.50.1`、`qwen-vl-utils`、`pandas`、`pyarrow`、`httpx[socks]`。

### 项目代码

- 已新增 `vlm_pairwise.py`：
  - 读取街景图片目录；可随机生成图片对，或读取固定配对 CSV。
  - 使用 Qwen2.5-VL 3B + 4-bit NF4。
  - 题目固定为「哪张更干净／视觉污染更低」，所以写入 `winner` 的图片与现有 `main.py` 的 VP 反向逻辑一致。
  - 输出 `image_a,image_b,winner` 及模型、提示词、时间戳、原始回答和状态字段。
- `README.md` 已新增本地 4-bit VLM 的运行说明。
- 已通过：`conda run -n guvp-vlm python -m py_compile vlm_pairwise.py` 与 `python vlm_pairwise.py --help`。

### 已下载的 Global Streetscapes 元数据

Hugging Face 缓存根目录：
`/home/colin/.cache/huggingface/hub/datasets--NUS-UAL--global-streetscapes/snapshots/7bd2e7697a3cb5f74ff05bd718babdb927f8b60d/`

已完成：

- `data/parquet/metadata_common_attributes.parquet`（530MB）
- `data/parquet/metadata_kv.parquet`（118MB）
- `data/parquet/streetscapes.parquet`（约 1.98GB；下载完成的依据是缓存约 2.5GB 且无 `.incomplete` 文件。下一步应先确认快照路径存在。）
- `info.csv`

官方数据说明：聚合表应提供所需关联字段，包括 `uuid`、`source`、`orig_id`、`lat`、`lon`、`datetime_local`、`sequence_id`、`city`、`quality`、`platform`、道路/空间相关字段等。需要用 `pyarrow` 的 schema 核对实际列名。

## 尚未完成 / 当前下一步

1. 确认 `streetscapes.parquet` 的快照文件路径与字段。
2. 新建并运行可复现的筛选脚本（建议名：`prepare_global_streetscapes_sample.py`）：
   - 城市 = Amsterdam、Paris、Berlin、Barcelona、Rome；
   - 只保留 `source == KartaView` 作为第一轮无 token 候选；
   - 保留室外 / `platform` 合格、`quality == good`、白天、无明显 glare/reflection 的记录（以真实字段值为准）；
   - 每个 `sequence_id` 仅保留一张，避免相邻视角；
   - 按经纬度网格分层抽样，避免集中于市中心；
   - 目标 50 张/城市，固定随机种子；
   - 输出 `data/images_manifest.csv`，必须保留 `uuid,source,orig_id,city,lat,lon,datetime_local,sequence_id`、质量/道路字段、网格单元、筛选规则与随机种子。
3. 先报告 KartaView 每城候选数。若任何城市不足 50，才扩展到 Mapillary。
4. 下载图片：Mapillary 来源需要用户在本机安全配置免费的 Mapillary access token；不要要求用户在聊天中粘贴 token。KartaView 候选可直接下载。官方脚本要求输入至少含 `uuid,source,orig_id`。
5. 图片下载完成后，运行：

```bash
conda activate guvp-vlm
python vlm_pairwise.py --images-dir data/street_images --max-pairs 20 --output data/vlm_pairwise_votes.csv
python main.py --input data/vlm_pairwise_votes.csv --output-dir output/vlm --dimension visual_pollution
```

## Qwen 模型下载

- 已启动 `Qwen/Qwen2.5-VL-3B-Instruct` 的 Hugging Face 下载；此前状态为约 5.1GB 缓存、两个权重分片仍为 `.incomplete`，因此当时还不可加载完整模型。
- 新会话应先检查：

```bash
pgrep -af 'snapshot_download.*Qwen2.5-VL-3B'
find /home/colin/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct -type f -name '*.incomplete'
```

- 如下载已完成，先用两张实际街景图做一次 4-bit 推理 smoke test；不要使用流程图或合成图作为研究样本。

## 重要限制

- Global Streetscapes 的 Hugging Face 仓库托管的是表格元数据，不托管全量 JPEG。
- 官方图像下载需要根据 `uuid/source/orig_id` 从 Mapillary/KartaView API 取回；Mapillary 需要 access token。
- 当前项目不是 Git 仓库，因此不能依靠 `git diff` 管理改动。
