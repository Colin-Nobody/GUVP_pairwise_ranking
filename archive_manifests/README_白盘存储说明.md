# GUVP 外置存储说明

归档日期：2026-09-20

## 目录结构

- `huggingface_cache/hub/`
  - `models--Qwen--Qwen2.5-VL-3B-Instruct/`：Qwen2.5-VL-3B 完整 Hugging Face 缓存。
  - `models--Qwen--Qwen3-VL-8B-Instruct/`：Qwen3-VL-8B 完整 Hugging Face 缓存。
  - `datasets--NUS-UAL--global-streetscapes/`：Global Streetscapes 全量 Parquet 与元数据缓存。
- `project_backups/GUVP_pairwise_ranking_2026-09-20/`
  - 当前项目完整快照，包含代码、250 张 JPEG、`images_manifest.csv`、视觉污染监督分数、分割计划与历史实验结果。
- `downloads_archive/`
  - `documents/`：PDF 文档。
  - `workflows/`：Dify 插件包与工作流 YAML。
  - `images/`：图片资料。
  - `software_resources/`：软件资源压缩包。
  - `project_notes/`：项目记忆和交接资料。

## 日常使用

系统盘原 Hugging Face 缓存位置已经建立符号链接，保持本硬盘挂载在 `/media/colin/白` 时，原有模型调用与 Global Streetscapes 路径可以继续使用。

当前活跃项目仍位于 `/home/colin/GUVP_pairwise_ranking`。外置盘中的项目目录是备份快照，不建议直接作为日常开发目录。

分割与回归流程应继续使用系统盘上的活跃项目：

1. 从 `data/street_images/` 读取 250 张图片。
2. 通过 `data/images_manifest.csv` 按 UUID 连接元数据。
3. 运行分割模型，生成图像级语义特征。
4. 与 `output/vlm_7b_definition_v8_3000_prefix_cache/perception_scores.csv` 中的视觉污染分数连接。
5. 使用 Elastic Net 等回归模型训练。

## 注意事项

- Qwen2.5-VL-7B 因近期仍在使用，保留在系统盘，未迁移到这里。
- Conda 环境和当前分割环境没有迁移。
- 使用3B、8B或全量 Global Streetscapes 数据前，必须先连接并挂载“白”盘。
- 250张图片和监督数据在系统盘保留工作副本，在此处保留备份副本。
- 重要结果不要只保存在这块机械硬盘上；建议继续保留至少另一份副本。
