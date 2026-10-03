# VP Object 提示词与双模型位置复核

## 最终提示词

采用标签前置布局：

```text
Image A:
[image A]
Image B:
[image B]
Compare the two street-view images fairly. Visual pollution means visible intrusive or disordered elements such as excessive signs or advertising, tangled overhead wires and poles, litter or debris, damaged or neglected surfaces, and redundant street clutter. Which image has LOWER overall visual pollution and looks cleaner? Return exactly: A or B.
```

该版本只在已经验证稳定的极短提示词中补充少量、可直接观察的 object 例子，不加入文化背景、主观心理量表、分项权重、逐图打分过程或 `T` 选项。

## 实验设计

- 160 个自然图片对，每对运行 AB 和 BA，共 320 次判断；
- 40 个相同图片控制；
- 20 个 TARGET 视觉访问控制，每个运行 AB 和 BA，共 40 次；
- 推理采用确定性解码；
- 只有 AB 与 BA 指向同一张实际图片，才视为稳定判断；
- 后续模型限定为 Qwen2.5-VL-7B-Instruct 和 Qwen3-VL-8B-Instruct，排除 3B。

## 结果

| 模型与提示 | 自然图 A 率 | AB/BA 换位一致率 | TARGET 准确率 |
|---|---:|---:|---:|
| 7B，原始极短提示 | 50.3% | 86.9% | 100.0% |
| 7B，object 提示 | 50.9% | 88.1% | 100.0% |
| Qwen3 8B，原始极短提示 | 55.9% | 80.6% | 100.0% |
| Qwen3 8B，object 提示 | 60.6% | 76.2% | 100.0% |

使用 object 提示后，两模型在共同换位稳定的 111 个自然图片对中，有 98 对选择同一张实际图片作为低 VP 图片，一致率为 88.3%。原始提示对应的一致率为 86.0%。

## 决策

- 采用上述 object 提示作为下一阶段提示词；
- 7B 与 Qwen3 8B 分别独立运行、独立保存 stable votes、独立计算排名；
- 正式清单包含 5,188 个均衡无序图片对，每张图参与 41 或 42 对；AB/BA 后每个模型执行 10,376 次方向推理；
- 每个无序图片对必须运行 AB 和 BA；
- 单模型内部换位不一致的图片对不计票，单独记录为位置敏感或不确定；
- 两套正式排名完成后再进行跨模型一致性分析；跨模型结果只用于复核，不混入任一模型的计票和排名。

原始逐次输出和机器可读汇总位于 `output/vlm_prompt_v5_audit/`，提示词文件为 `prompts/vp_pairwise_object_cues_v5.txt`。
