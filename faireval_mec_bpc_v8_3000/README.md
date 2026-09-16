# GUVP Definition Prompt V8 — 3,000-pair MEC+BPC run

This self-contained archive records the Qwen2.5-VL-7B experiment completed on
2026-09-12. It includes the inference and ranking code, exact V7/V8 prompts,
balanced pair manifest, accepted votes, complete raw-response audit, final
250-image ranking, and a paired V7-vs-V8 sensitivity analysis.

## Experiment

- 250 street-view images and 3,000 unique unordered pairs.
- Every image appears in exactly 24 comparisons.
- Balanced-position calibration: every pair is evaluated as AB and BA.
- Multi-evidence calibration: three sampled evaluations per position.
- 18,000 directional samples in total.
- V8 states that a lower score means lower visual pollution.
- The lower calibrated score becomes the pairwise `winner` (the cleaner image).
- Q-score and TrueSkill convert accepted cleaner-image preferences into the final ranking.

See [`MODEL_CONFIG.json`](MODEL_CONFIG.json) for the frozen configuration and
[`RESULTS.md`](RESULTS.md) for headline results.

## Reproduction

Place the same 250 images in `data/street_images/`, install the dependencies, and run:

```bash
python -m pip install -r requirements.txt -r requirements-vlm.txt
./scripts/run_7b_definition_v8_3000_and_score.sh
```

The exact prompt is [`prompts/vp_pairwise_definition_v8.txt`](prompts/vp_pairwise_definition_v8.txt).

## Key artifacts

- Accepted votes: [`data/vlm_pairwise_7b_definition_v8_3000_prefix_cache.csv`](data/vlm_pairwise_7b_definition_v8_3000_prefix_cache.csv)
- Complete audit: [`results/7b_formal/pair_audit.csv`](results/7b_formal/pair_audit.csv)
- Final ranking: [`results/7b_formal/perception_scores.csv`](results/7b_formal/perception_scores.csv)
- V7/V8 comparison: [`results/v7_3000_vs_v8_3000/consistency_summary.json`](results/v7_3000_vs_v8_3000/consistency_summary.json)

The archived `winner` is the lower-pollution image. Do not merge these votes
with runs using a different prompt or score direction.
