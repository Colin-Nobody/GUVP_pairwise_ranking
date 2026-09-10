# GUVP Definition Prompt V7 — 1,500-pair MEC+BPC run

This directory is a self-contained archive of the Qwen2.5-VL-7B visual-pollution
experiment completed on 2026-09-10. It contains the inference/ranking code,
deterministic balanced pair manifest, exact prompt, raw audit responses, accepted
votes, final 250-image TrueSkill ranking, and the comparison with the earlier V6
3,000-pair run.

## Experiment

- 250 street-view images.
- 1,500 unique unordered image pairs; every image appears exactly 12 times.
- Balanced-position calibration (BPC): every physical pair is evaluated as AB and BA.
- Multi-evidence calibration (MEC): three sampled evaluations per position.
- 9,000 directional samples in total.
- A visual-pollution score of 1 means lower VP and 10 means higher VP.
- The lower calibrated VP score is written as the pairwise `winner` (the cleaner image).
- Q-score and TrueSkill convert accepted lower-VP preferences into the final ranking.

The complete frozen configuration is in [`MODEL_CONFIG.json`](MODEL_CONFIG.json).

## Prompt

The exact prompt is [`prompts/vp_pairwise_definition_v7.txt`](prompts/vp_pairwise_definition_v7.txt).
Its runtime SHA-256 after whitespace normalization is
`cd3722b5e318415a1b2fbb266c8e898d22c2a0c45939b72dcb07d86fae8a23bc`.

## Reproduction

Create `data/street_images/` containing the same 250 input images, install the
dependencies, and run:

```bash
python -m pip install -r requirements.txt -r requirements-vlm.txt
./scripts/run_7b_definition_v7_1500_and_score.sh
```

Monitor the run from another terminal:

```bash
./scripts/monitor_7b_definition_v7_1500.sh 10
```

The runner performs a preflight check before loading the model, records the prompt
hash and score direction, supports audit-based resume, summarizes the MEC+BPC
outcomes, and generates the final ranking.

## Results

See [`RESULTS.md`](RESULTS.md) for headline results and interpretation.

- Accepted votes: [`data/vlm_pairwise_7b_definition_v7_1500_prefix_cache.csv`](data/vlm_pairwise_7b_definition_v7_1500_prefix_cache.csv)
- Raw pair audit: [`results/7b_formal/pair_audit.csv`](results/7b_formal/pair_audit.csv)
- Final ranking: [`results/7b_formal/perception_scores.csv`](results/7b_formal/perception_scores.csv)
- V6/V7 comparison: [`results/v6_3000_vs_v7_1500/comparison_summary.json`](results/v6_3000_vs_v7_1500/comparison_summary.json)

Do not merge the V6 and V7 vote files: their prompts and score semantics differ.
