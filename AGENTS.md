# GUVP：Codex 项目说明

## 新会话恢复

每次在本仓库开始新会话，先读取根目录 `CODEX_CONTEXT.md`，获取用户目标、已经确认的研究决策、实施进度和下一步。

在设计、实现、运行或评价 ViT/DINOv3→视觉污染分数实验之前，完整读取 `VIT_VP_EXPERIMENT_PLAN.md`。这是本路线的详细方法依据；不要依靠历史聊天记忆补全参数。

若涉及旧分割流程，再读取 `SEGMENTATION_PLAN.md`、`SEGMENTATION_EXPERIMENT_STATUS.md`；若涉及原 Elastic Net 方法，再读取 `VP_POLLUTION_ELASTIC_NET_LOGBOOK.md`。这些文档描述的旧目标和样本规模不能覆盖 ViT 计划中已经核实的 1,000 图设置。

## 已确定的研究方向

用户希望研究 `image → DINOv3 ViT-B/16 → prediction head → visual_pollution_score_0_10`，用单图直接预测替代部署时的 segmentation→Elastic Net 流程。研究当前有 1,000 张图片和对应 VP 标签，不要继续假定只有 250 张。

## 计算设备约束

- 所有 VLM/ViT 模型推理及训练均须在 CUDA GPU 上执行。禁止 CPU offload、CPU fallback 和未验证可能将模型层放到 CPU 的自动 device map；CUDA 不可用或模型不能完整放入 GPU 时应直接报错退出。
- CPU 可用于图片解码、数据预处理、文件 I/O 和非模型统计计算；不得用 CPU 执行模型 forward/generate。
- 新增或修改模型入口、作业脚本时，必须保留 GPU 可用性和最终设备映射检查。资源不足时调整 GPU 资源或模型配置，不得静默改用 CPU。

模型对照、标签版本、城市分组、比较记录隔离、内外层交叉验证、尺度和指标规则以完整计划为准。任何方法变更须记录理由及影响；用户后续明确的新要求优先，并同步更新文档。

## 工作与交接

- 开始实施前核对实际文件、代码和运行状态；交接记录中的事实带有核实日期。
- 实验输出放入计划指定的新目录，保留现有数据、评分、训练环境和实验成果；保留用户未提交的修改。
- 严格区分计划、代码实现、校验、任务提交、运行中和已完成；没有真实输出时不能标记阶段完成。
- 每次完成一段相关工作后更新 `CODEX_CONTEXT.md`：已完成事项、产物路径、运行/提交信息（如有）、检查结果、未解决问题和下一步。阶段状态只维护在该交接文件，避免多处状态互相矛盾。
- 若研究方案发生变化，同时更新 `VIT_VP_EXPERIMENT_PLAN.md` 的对应方法，记录日期和原因。大文件日志放入产物目录，交接文件保留简短索引。
- 当前用户请求的范围是持久化研究计划；后续执行范围由后续用户请求及已有授权决定。新会话读取这些文件不会自行授权下载、安装或启动训练。

## 读取机制说明

Codex 的项目入口为 `AGENTS.md`；`CODEX_CONTEXT.md` 和完整计划通过这里的明确阅读要求发现。

本说明按 OpenAI 官方文档的项目指令机制组织：
https://developers.openai.com/codex/guides/agents-md
