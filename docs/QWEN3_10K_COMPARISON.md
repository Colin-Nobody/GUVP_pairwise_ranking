# Qwen3-VL-8B vs Qwen2.5-VL-7B: 20-city V8 comparison

The final Qwen3 job 388577 processed 65,000 unordered pairs over 10,000 images, using 3 MEC samples per image order (390,000 directional samples), and BPC position swapping. It produced 64,142 calibrated non-tie votes, 795 calibrated ties and 63 pairs with invalid samples (30 AB, 33 BA). Model: local pre-quantized Qwen3-VL-8B-Instruct NF4; prompt version vp-definition-score-v8, SHA-256 718e5c4dac829a70e4eb1dab07743a0c7869b3b90ebda6b23b7b464c205acf5b. The paired Qwen2.5-VL-7B experiment had 57,260 calibrated votes from 65,000 pairs. The full audits' ordered image_a/image_b columns are identical row-by-row.

Qwen3 internal checks:
- Across 10,000 images, Q-score versus TrueSkill mu: Spearman 0.9924430352992908; Pearson 0.9912147355031167. These aggregations use the same pairwise data; agreement is not external validation.
- Of 64,937 image pairs with 3 parsed samples in each direction, align scores to physical image identities, sum each image's three scores within each order, and subtract image B from A: AB vs BA margin Spearman 0.8405352948647044; Pearson 0.8128182475832922. 53,537 margins agree in sign, 10,484 disagree, and 916 have a zero margin in at least one direction. Sign agreement excluding zero margins: 53,537 / 64,021 = 0.8362412333453085. This is an order-consistency statistic including stochastic sampling variation, not a direct pure position-bias estimate.

Cross-model comparison:
- Full audits have the same 65,000 input pairs; both models produced valid non-tie votes for 56,565 common pairs. They chose the same physical winner for 44,193 pairs: 78.1278%.
- All 10,000 images matched. Their normalized 0–10 VP scores have Spearman rho 0.7581900691257446, Pearson r 0.7598522418149652 and Kendall tau 0.56316303514634. Mean absolute rank difference is 1,513.268 places, and mean absolute normalized VP score difference is 1.2162773025.
- These numbers are comparisons between models, not accuracy, agreement with people, or independent validation.
- The older general comparison script labels one plot as A/B position selection. Its counts use the original image_a after bidirectional calibration; do not interpret this plot as raw position bias.

Source result directories are output/vlm_qwen3_10k_v8_pairbatch50_mec3/ and output/qwen_compare/result/. Large CSVs containing raw model responses use Git LFS; run git lfs pull to retrieve actual data. The input Qwen2.5 results are on hpc-10k-vp-results. Images and model weights are not included in this branch's new result files.
