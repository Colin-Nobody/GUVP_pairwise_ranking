# DeepLabV3+ Mapillary Vistas v2.0 目标域审计

## 状态

Mapillary 训练后模型的 30 图分层审计和两种推理实验已经完成。人工评分仍需
在 `visual_review/index.html` 中完成。

## 模型

- 配置：`/home/colin/GUVP_pairwise_ranking/segmentation/mapillary_vistas/configs/deeplabv3plus_r50-d8_1xb2-80k_mapillary-v2-512x512.py`
- checkpoint：`/home/colin/GUVP_pairwise_ranking/output/deeplabv3plus_r50_mapillary_v2_80k/best_mIoU_iter_80000.pth`
- 类别数：124
- 审计图像：30（每城市高/中/低 VP 各 2 张）

## 完整性检查

- baseline：30 个原尺寸 mask、30 个 overlay、
  3720 行类别统计。
- sliding：30 个原尺寸 mask、30 个 overlay、
  3720 行类别统计。
- 每张图的类别像素比例之和均为 1；不存在缺失输出。
- baseline 总推理时间 2.33 秒，GPU 峰值
  819.7 MiB。
- sliding 总推理时间 191.96 秒，GPU 峰值
  449.9 MiB。

## 方法敏感性

- 平均像素一致率：0.7258
- 中位像素一致率：0.7651
- 一致率范围：0.2108--0.9626
- 方法间平均类别 IoU：0.2103
- 平均检测类别数：baseline 16.40，sliding
  24.33

以上比较衡量两种推理策略的一致程度，不代表真实准确率。最终方法必须结合
人工审核决定。

## 人工审核

打开 `visual_review/index.html`，评分标准：0=不可用或严重错误，1=部分可用，
2=总体可靠，NA=无法判断。重点检查广告牌、店招、道路、人行道、杆件、路灯、
垃圾桶、坑洞、临时护栏、车辆和植被等 Mapillary 类别，并记录滑窗接缝、
小目标遗漏和类别混淆。
