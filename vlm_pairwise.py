"""Run FairEval-style MEC+BPC pairwise evaluation for urban visual pollution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
}
PROMPT_VERSION = "vp-faireval-mec-bpc-score-v6"
LABEL_LAYOUT = "label_before"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "vp_pairwise_faireval_mec_bpc_v6.txt"
PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
QUANTIZATION = "bitsandbytes-nf4-double-bf16-if-supported-else-fp16"
DECODE_MODE = "sampled_evidence_first_scores"

AUDIT_FIELDS = [
    "comparison_id", "image_a", "image_b", "winner",
    "calibrated_score_a", "calibrated_score_b", "calibrated_margin",
    "scores_ab_image_a", "scores_ab_image_b", "scores_ba_image_a", "scores_ba_image_b",
    "valid_samples_ab", "valid_samples_ba", "bpde",
    "model", "prompt_version", "prompt_sha256", "label_layout", "max_pixels",
    "quantization", "cpu_offload", "decode_mode", "mec_samples", "temperature",
    "max_new_tokens", "seed", "timestamp_utc", "raw_responses_ab", "raw_responses_ba", "status",
]
VOTE_FIELDS = list(AUDIT_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, help="Optional CSV containing image_a and image_b")
    parser.add_argument("--output", type=Path, default=Path("data/vlm_pairwise_votes.csv"),
                        help="Calibrated non-tie votes; directly accepted by main.py")
    parser.add_argument("--audit-output", type=Path,
                        help="All MEC+BPC outcomes; defaults to OUTPUT with _audit before the suffix")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_PATH,
                        help="Prompt text file (default: the v6 FairEval prompt)")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION,
                        help="Version label recorded in audit output")
    parser.add_argument(
        "--score-direction",
        choices=("higher_is_cleaner", "lower_is_cleaner"),
        default="higher_is_cleaner",
        help="How model scores are converted to the lower-visual-pollution winner",
    )
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--limit-pairs", type=int,
                        help="Optional prefix limit for a fixed --pairs manifest; intended for smoke tests")
    parser.add_argument("--all-pairs", action="store_true", help="Compare every unique unordered image pair")
    parser.add_argument("--resume", action="store_true", help="Resume from complete audit rows")
    parser.add_argument("--overwrite", action="store_true", help="Replace vote and audit outputs")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="Report every N completed unordered pairs")
    parser.add_argument("--progress-file", type=Path, help="Optional append-only progress log")
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Deprecated and rejected: project inference requires GPU-only execution")
    parser.add_argument("--gpu-only", action="store_true", default=True,
                        help="Compatibility flag; GPU-only execution is enforced for every run")
    parser.add_argument("--mec-samples", type=int, default=3,
                        help="Evidence-first samples per position (paper default: 3)")
    parser.add_argument("--mec-batch-size", type=int, default=1,
                        help="MEC samples decoded together; 1 preserves the serial baseline")
    parser.add_argument("--reuse-prefix-cache", action="store_true",
                        help="Reuse one visual/prompt KV prefix across serial MEC samples")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="MEC sampling temperature (paper default: 1.0)")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing-output", type=Path,
                        help="Optional JSON timing report for controlled benchmarks")
    return parser.parse_args()


def inference_decode_mode(args: argparse.Namespace) -> str:
    batch_size = getattr(args, "mec_batch_size", 1)
    if getattr(args, "reuse_prefix_cache", False):
        mode = f"{DECODE_MODE}_serial_reused_prefix_multinomial"
    elif batch_size == 1:
        mode = DECODE_MODE
    else:
        mode = f"{DECODE_MODE}_mec_batch{batch_size}_independent_multinomial"
    if getattr(args, "score_direction", "higher_is_cleaner") == "lower_is_cleaner":
        mode += "_lower_score_is_cleaner"
    return mode


def default_audit_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_audit{output.suffix}")


def load_prompt(path: Path) -> tuple[str, str]:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def get_images(directory: Path) -> dict[str, Path]:
    files = {
        item.name: item for item in sorted(directory.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    }
    if len(files) < 2:
        raise ValueError(f"Need at least two images in {directory}; found {len(files)}")
    return files


def get_pairs(args: argparse.Namespace, files: dict[str, Path]) -> list[tuple[str, str]]:
    if args.pairs:
        with args.pairs.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"image_a", "image_b"}.issubset(reader.fieldnames):
                raise ValueError("--pairs must contain image_a,image_b columns")
            pairs = [(row["image_a"].strip(), row["image_b"].strip()) for row in reader]
        if args.limit_pairs is not None:
            if args.limit_pairs < 1:
                raise ValueError("--limit-pairs must be positive")
            pairs = pairs[:args.limit_pairs]
    else:
        all_pairs = list(itertools.combinations(files, 2))
        order = torch.randperm(
            len(all_pairs), generator=torch.Generator().manual_seed(args.seed)
        ).tolist()
        selected = order if args.all_pairs else order[:args.max_pairs]
        pairs = [all_pairs[item] for item in selected]

    if not pairs:
        raise ValueError("No image pairs to compare")
    seen: set[frozenset[str]] = set()
    for image_a, image_b in pairs:
        if image_a == image_b or image_a not in files or image_b not in files:
            raise ValueError(f"Invalid image pair: {image_a!r}, {image_b!r}")
        key = frozenset((image_a, image_b))
        if key in seen:
            raise ValueError(f"Duplicate unordered image pair: {image_a!r}, {image_b!r}")
        seen.add(key)
    return pairs


def _score_match(response: str, label: str) -> re.Match[str] | None:
    # Qwen sometimes wraps the otherwise correct score line in Markdown, e.g.
    # ``**Score of Image A:** 4``. Treat that as formatting, not a bad score.
    pattern = (
        rf"(?im)^\s*\*{{0,2}}\s*(?:the\s+)?score\s+of\s+image\s+{label}"
        rf"\s*:\s*\*{{0,2}}\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*\*{{0,2}}\s*$"
    )
    return re.search(pattern, response)


def parse_evaluation(response: str) -> tuple[float, float] | None:
    """Parse an evidence-first response, rejecting missing evidence and out-of-range scores."""
    evidence = re.search(
        r"(?im)^\s*\*{0,2}\s*evaluation evidence\s*:\s*\*{0,2}\s*(\S.*)$",
        response,
    )
    match_a = _score_match(response, "A")
    match_b = _score_match(response, "B")
    if not evidence or not match_a or not match_b:
        return None
    if evidence.start() > min(match_a.start(), match_b.start()):
        return None
    score_a, score_b = float(match_a.group(1)), float(match_b.group(1))
    if not (1.0 <= score_a <= 10.0 and 1.0 <= score_b <= 10.0):
        return None
    return score_a, score_b


def physical_winner(image_a: str, image_b: str, choice: str | None) -> str | None:
    """Compatibility helper retained for older audit utilities/tests."""
    return image_a if choice == "A" else image_b if choice == "B" else None


def classify_pair(
    original_a: str, original_b: str, choice_ab: str | None, choice_ba: str | None,
) -> tuple[str | None, str | None, str | None, str]:
    """Legacy strict AB/BA classifier; MEC+BPC uses calibrate_scores instead."""
    winner_ab = physical_winner(original_a, original_b, choice_ab)
    winner_ba = physical_winner(original_b, original_a, choice_ba)
    if winner_ab is None and winner_ba is None:
        status = "invalid_both"
    elif winner_ab is None:
        status = "invalid_ab"
    elif winner_ba is None:
        status = "invalid_ba"
    elif winner_ab != winner_ba:
        status = "swap_inconsistent"
    else:
        status = "stable"
    return winner_ab, winner_ba, winner_ab if status == "stable" else None, status


def _outcome(score_a: float, score_b: float) -> str:
    return "win" if score_a > score_b else "loss" if score_a < score_b else "tie"


def diversity_entropy(outcomes: list[str]) -> float | None:
    """Balanced-position diversity entropy (natural logarithm), useful for HITL triage."""
    if not outcomes:
        return None
    return -sum(
        (outcomes.count(label) / len(outcomes)) * math.log(outcomes.count(label) / len(outcomes))
        for label in ("win", "tie", "loss") if outcomes.count(label)
    )


def calibrate_scores(
    image_a: str,
    image_b: str,
    evaluations_ab: list[tuple[float, float]],
    evaluations_ba: list[tuple[float, float]],
    expected_samples: int,
    score_direction: str = "higher_is_cleaner",
) -> dict[str, object]:
    """Realign scores to physical images, then average k samples across both positions."""
    if score_direction not in {"higher_is_cleaner", "lower_is_cleaner"}:
        raise ValueError(f"Unsupported score direction: {score_direction}")
    preference_sign = 1.0 if score_direction == "higher_is_cleaner" else -1.0
    scores_ab_a = [item[0] for item in evaluations_ab]
    scores_ab_b = [item[1] for item in evaluations_ab]
    scores_ba_a = [item[1] for item in evaluations_ba]
    scores_ba_b = [item[0] for item in evaluations_ba]
    outcomes = [_outcome(preference_sign * a, preference_sign * b) for a, b in evaluations_ab]
    outcomes += [
        _outcome(preference_sign * a, preference_sign * b) for b, a in evaluations_ba
    ]
    entropy = diversity_entropy(outcomes)

    result: dict[str, object] = {
        "winner": "", "calibrated_score_a": "", "calibrated_score_b": "",
        "calibrated_margin": "", "scores_ab_image_a": json.dumps(scores_ab_a),
        "scores_ab_image_b": json.dumps(scores_ab_b), "scores_ba_image_a": json.dumps(scores_ba_a),
        "scores_ba_image_b": json.dumps(scores_ba_b), "valid_samples_ab": len(evaluations_ab),
        "valid_samples_ba": len(evaluations_ba), "bpde": "" if entropy is None else entropy,
    }
    valid_ab = len(evaluations_ab) == expected_samples
    valid_ba = len(evaluations_ba) == expected_samples
    if not valid_ab or not valid_ba:
        result["status"] = (
            "invalid_both" if not valid_ab and not valid_ba
            else "invalid_ab" if not valid_ab else "invalid_ba"
        )
        return result

    calibrated_a = (sum(scores_ab_a) + sum(scores_ba_a)) / (2 * expected_samples)
    calibrated_b = (sum(scores_ab_b) + sum(scores_ba_b)) / (2 * expected_samples)
    # Positive margin always means physical image A has lower visual pollution.
    margin = preference_sign * (calibrated_a - calibrated_b)
    result.update({
        "calibrated_score_a": calibrated_a,
        "calibrated_score_b": calibrated_b,
        "calibrated_margin": margin,
        "winner": image_a if margin > 0 else image_b if margin < 0 else "",
        "status": "calibrated" if margin != 0 else "calibrated_tie",
    })
    return result


class IndependentMultinomialSampler(LogitsProcessor):
    """Select one token per batch row with the baseline multinomial sampler."""

    def __init__(self, seeds: list[int], temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("IndependentMultinomialSampler requires a positive temperature")
        self.seeds = seeds
        self.temperature = temperature
        self.generators: list[torch.Generator] | None = None
        self.generator_device: torch.device | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != len(self.seeds):
            raise ValueError(
                f"Expected {len(self.seeds)} batch rows, received {scores.shape[0]}"
            )
        if self.generators is None or self.generator_device != scores.device:
            self.generators = [
                torch.Generator(device=scores.device).manual_seed(seed) for seed in self.seeds
            ]
            self.generator_device = scores.device
        probabilities = torch.softmax(scores / self.temperature, dim=-1)
        selected = [
            torch.multinomial(row, num_samples=1, generator=generator)
            for row, generator in zip(probabilities, self.generators, strict=True)
        ]
        selected_tokens = torch.cat(selected).unsqueeze(1)
        forced_scores = torch.full_like(scores, -torch.inf)
        forced_scores.scatter_(1, selected_tokens, 0.0)
        return forced_scores


def prepare_pair_inputs(
    processor, image_a: Path, image_b: Path, max_pixels: int, prompt: str = PROMPT,
):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Image A:"},
        {"type": "image", "image": str(image_a), "max_pixels": max_pixels},
        {"type": "text", "text": "Image B:"},
        {"type": "image", "image": str(image_b), "max_pixels": max_pixels},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )


def repeat_inputs(inputs, batch_size: int, device: torch.device) -> dict[str, object]:
    """Repeat one complete multimodal prompt, preserving each [A, B] visual block."""
    repeated: dict[str, object] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            repeats = (batch_size,) + (1,) * (value.ndim - 1)
            repeated[key] = value.repeat(repeats).to(device)
        elif isinstance(value, list):
            repeated[key] = value * batch_size
        else:
            repeated[key] = value
    return repeated


def generate_prepared(
    model, processor, inputs, temperature: float, max_new_tokens: int, seeds: list[int],
    independent_sampling: bool,
) -> list[str]:
    batched = repeat_inputs(inputs, len(seeds), model.device)
    sampling = temperature > 0
    generation_args: dict[str, object] = {"max_new_tokens": max_new_tokens, "do_sample": sampling}
    if sampling:
        if independent_sampling:
            generation_args["do_sample"] = False
            generation_args["logits_processor"] = LogitsProcessorList([
                IndependentMultinomialSampler(seeds, temperature)
            ])
        else:
            generation_args["temperature"] = temperature
            torch.manual_seed(seeds[0])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seeds[0])
    with torch.inference_mode():
        generated = model.generate(**batched, **generation_args)
    responses = processor.batch_decode(
        generated[:, batched["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return [response.strip() for response in responses]


def generate_with_reused_prefix(
    model, processor, inputs, temperature: float, max_new_tokens: int, seeds: list[int],
) -> list[str]:
    """Encode the multimodal prefix once, then decode each MEC sample serially.

    The first token is sampled explicitly so the remaining generation can start
    from the already-computed KV cache. The cache is cropped back to the prompt
    length between samples, preserving baseline batch-size-one arithmetic.
    """
    prepared = repeat_inputs(inputs, 1, model.device)
    prompt_ids = prepared["input_ids"]
    prompt_length = prompt_ids.shape[1]
    with torch.inference_mode():
        prefix_output = model(**prepared, use_cache=True, logits_to_keep=1)
    prefix_cache = prefix_output.past_key_values
    prefix_logits = prefix_output.logits[:, -1].to(dtype=torch.float32)
    if not hasattr(prefix_cache, "crop"):
        raise TypeError("The model cache does not support prefix reuse via crop()")

    repetition_penalty = float(model.generation_config.repetition_penalty or 1.0)
    repetition_processor = (
        RepetitionPenaltyLogitsProcessor(repetition_penalty)
        if repetition_penalty != 1.0 else None
    )
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    responses: list[str] = []
    for seed in seeds:
        cached_length = prefix_cache.get_seq_length()
        if cached_length < prompt_length:
            raise ValueError(
                f"Prefix cache shrank below the prompt length: {cached_length} < {prompt_length}"
            )
        if cached_length > prompt_length:
            prefix_cache.crop(-(cached_length - prompt_length))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        first_scores = prefix_logits.clone()
        if repetition_processor is not None:
            first_scores = repetition_processor(prompt_ids, first_scores)
        if temperature > 0:
            first_probabilities = torch.softmax(first_scores / temperature, dim=-1)
            first_token = torch.multinomial(first_probabilities, num_samples=1)
        else:
            first_token = torch.argmax(first_scores, dim=-1, keepdim=True)
        generated_prefix = torch.cat([prompt_ids, first_token], dim=1)

        if first_token.item() in eos_ids or max_new_tokens == 1:
            generated = generated_prefix
        else:
            attention_mask = torch.cat(
                [prepared["attention_mask"], torch.ones_like(first_token)], dim=1
            )
            generation_args: dict[str, object] = {
                "max_new_tokens": max_new_tokens - 1,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generation_args["temperature"] = temperature
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=generated_prefix,
                    attention_mask=attention_mask,
                    past_key_values=prefix_cache,
                    **generation_args,
                )
        response = processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )[0].strip()
        responses.append(response)
    return responses


def infer_once(
    model, processor, image_a: Path, image_b: Path, max_pixels: int,
    temperature: float, max_new_tokens: int, sample_seed: int,
    timings: dict[str, float] | None = None,
    prompt: str = PROMPT,
) -> tuple[tuple[float, float] | None, str]:
    started = time.perf_counter()
    inputs = prepare_pair_inputs(processor, image_a, image_b, max_pixels, prompt)
    if timings is not None:
        timings["preprocess_seconds"] += time.perf_counter() - started
    started = time.perf_counter()
    response = generate_prepared(
        model, processor, inputs, temperature, max_new_tokens, [sample_seed],
        independent_sampling=False,
    )[0]
    if timings is not None:
        timings["generate_seconds"] += time.perf_counter() - started
    return parse_evaluation(response), response


def infer_mec(
    model, processor, image_a: Path, image_b: Path, args: argparse.Namespace, seed_offset: int,
    timings: dict[str, float] | None = None,
) -> tuple[list[tuple[float, float]], list[str]]:
    evaluations: list[tuple[float, float]] = []
    responses: list[str] = []
    prompt = getattr(args, "prompt", PROMPT)
    if getattr(args, "reuse_prefix_cache", False):
        started = time.perf_counter()
        inputs = prepare_pair_inputs(processor, image_a, image_b, args.max_pixels, prompt)
        if timings is not None:
            timings["preprocess_seconds"] += time.perf_counter() - started
        seeds = [args.seed + seed_offset + index for index in range(args.mec_samples)]
        started = time.perf_counter()
        responses = generate_with_reused_prefix(
            model, processor, inputs, args.temperature, args.max_new_tokens, seeds,
        )
        if timings is not None:
            timings["generate_seconds"] += time.perf_counter() - started
            timings["generation_batches"] += 1
        evaluations = [
            evaluation for response in responses
            if (evaluation := parse_evaluation(response)) is not None
        ]
        return evaluations, responses

    if args.mec_batch_size == 1:
        for sample_index in range(args.mec_samples):
            evaluation, response = infer_once(
                model, processor, image_a, image_b, args.max_pixels, args.temperature,
                args.max_new_tokens, args.seed + seed_offset + sample_index, timings, prompt,
            )
            responses.append(response)
            if evaluation is not None:
                evaluations.append(evaluation)
        return evaluations, responses

    started = time.perf_counter()
    inputs = prepare_pair_inputs(processor, image_a, image_b, args.max_pixels, prompt)
    if timings is not None:
        timings["preprocess_seconds"] += time.perf_counter() - started
    for first_index in range(0, args.mec_samples, args.mec_batch_size):
        stop_index = min(first_index + args.mec_batch_size, args.mec_samples)
        seeds = [args.seed + seed_offset + index for index in range(first_index, stop_index)]
        started = time.perf_counter()
        batch_responses = generate_prepared(
            model, processor, inputs, args.temperature, args.max_new_tokens, seeds,
            independent_sampling=True,
        )
        if timings is not None:
            timings["generate_seconds"] += time.perf_counter() - started
            timings["generation_batches"] += 1
        for response in batch_responses:
            responses.append(response)
            evaluation = parse_evaluation(response)
            if evaluation is not None:
                evaluations.append(evaluation)
    return evaluations, responses


def load_audit(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(AUDIT_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Audit file missing columns: {sorted(missing)}")
        return list(reader)


def validate_resume_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    expected = {
        "model": args.model,
        "prompt_version": getattr(args, "prompt_version", PROMPT_VERSION),
        "prompt_sha256": getattr(args, "prompt_sha256", PROMPT_SHA256),
        "label_layout": LABEL_LAYOUT, "max_pixels": str(args.max_pixels), "quantization": QUANTIZATION,
        "cpu_offload": str(args.cpu_offload).lower(), "decode_mode": inference_decode_mode(args),
        "mec_samples": str(args.mec_samples), "temperature": str(args.temperature),
        "max_new_tokens": str(args.max_new_tokens), "seed": str(args.seed),
    }
    for line_number, row in enumerate(rows, 2):
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"Audit row {line_number} uses incompatible {field}")
        key = (row.get("image_a", ""), row.get("image_b", ""))
        if key in completed:
            raise ValueError(f"Duplicate completed pair in audit row {line_number}: {key}")
        completed.add(key)
    return completed


def write_vote(writer: csv.DictWriter, row: dict[str, object]) -> None:
    writer.writerow({field: row[field] for field in VOTE_FIELDS})


def rebuild_votes(audit_rows: list[dict[str, str]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VOTE_FIELDS)
        writer.writeheader()
        for row in audit_rows:
            if row["status"] == "calibrated" and row["winner"]:
                write_vote(writer, row)
                accepted += 1
    return accepted


def load_model(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only inference is required, but CUDA is unavailable")
    if args.cpu_offload:
        raise ValueError("CPU offload is disabled by the project GPU-only policy")
    if torch.cuda.get_device_capability()[0] >= 8:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
    dtype_name = "bfloat16" if compute_dtype == torch.bfloat16 else "float16"
    device_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    print(
        f"GPU-only: {gpu_name} (compute capability {capability[0]}.{capability[1]}); "
        f"model compute dtype: {dtype_name}",
        flush=True,
    )
    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True, llm_int8_enable_fp32_cpu_offload=False,
    )
    device_map = device_index
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, quantization_config=config, dtype=compute_dtype, device_map=device_map,
        attn_implementation="sdpa",
    )
    device_map = getattr(model, "hf_device_map", None)
    gpu_devices = lambda device: isinstance(device, int) or (
        isinstance(device, str) and (device == "cuda" or device.startswith("cuda:"))
    )
    if not device_map or any(not gpu_devices(device) for device in device_map.values()):
        raise RuntimeError(f"GPU-only model placement check failed: {device_map}")
    print(f"GPU-only device placement verified across {len(device_map)} model modules", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    return model, processor


def validate_args(args: argparse.Namespace) -> None:
    if args.cpu_offload:
        raise ValueError("--cpu-offload is disabled by the project GPU-only policy")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if args.mec_samples < 1:
        raise ValueError("--mec-samples must be positive")
    mec_batch_size = getattr(args, "mec_batch_size", 1)
    if mec_batch_size < 1:
        raise ValueError("--mec-batch-size must be positive")
    if mec_batch_size > args.mec_samples:
        raise ValueError("--mec-batch-size cannot exceed --mec-samples")
    if getattr(args, "reuse_prefix_cache", False) and mec_batch_size != 1:
        raise ValueError("--reuse-prefix-cache requires --mec-batch-size 1")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.max_new_tokens < 16:
        raise ValueError("--max-new-tokens must be at least 16 for evidence-first output")


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    validate_args(args)
    args.prompt, args.prompt_sha256 = load_prompt(args.prompt_file)
    args.audit_output = args.audit_output or default_audit_path(args.output)
    if args.audit_output.resolve() == args.output.resolve():
        raise ValueError("--audit-output and --output must be different files")

    files = get_images(args.images_dir)
    pairs = get_pairs(args, files)
    indexed_pairs = [(f"vlm-{index:05d}", *pair) for index, pair in enumerate(pairs, 1)]

    audit_rows: list[dict[str, str]] = []
    if args.resume:
        if not args.audit_output.exists():
            raise ValueError(f"Cannot resume: audit output does not exist: {args.audit_output}")
        audit_rows = load_audit(args.audit_output)
        completed = validate_resume_rows(audit_rows, args)
    else:
        existing = [path for path in (args.output, args.audit_output) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(f"Output exists: {existing}. Use --overwrite, --resume, or new paths.")
        completed = set()

    pair_set = {(image_a, image_b) for _, image_a, image_b in indexed_pairs}
    unexpected = completed - pair_set
    if unexpected:
        raise ValueError(f"Audit contains {len(unexpected)} pair(s) absent from requested pair list")
    remaining = [item for item in indexed_pairs if (item[1], item[2]) not in completed]

    accepted = rebuild_votes(audit_rows, args.output)
    if not remaining:
        print(f"All {len(indexed_pairs)} unordered pairs are already complete.")
        print(f"Calibrated votes: {accepted}; audit: {args.audit_output}; votes: {args.output}")
        return

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    if args.progress_file:
        args.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with args.progress_file.open("a", encoding="utf-8") as handle:
            handle.write(
                f"Starting/resuming {len(indexed_pairs)} pairs; {len(completed)} complete; "
                f"prompt={args.prompt_version}; model={args.model}; MEC k={args.mec_samples}; "
                f"MEC batch={args.mec_batch_size}; temperature={args.temperature}; BPC=true; "
                f"score_direction={args.score_direction}\n"
            )

    load_started = time.perf_counter()
    model, processor = load_model(args)
    timings = {
        "model_load_seconds": time.perf_counter() - load_started,
        "preprocess_seconds": 0.0,
        "generate_seconds": 0.0,
        "generation_batches": 0.0,
    }
    audit_mode = "a" if args.resume else "w"
    with args.audit_output.open(audit_mode, newline="", encoding="utf-8") as audit_handle, \
            args.output.open("a", newline="", encoding="utf-8") as vote_handle:
        audit_writer = csv.DictWriter(audit_handle, fieldnames=AUDIT_FIELDS)
        vote_writer = csv.DictWriter(vote_handle, fieldnames=VOTE_FIELDS)
        if not args.resume:
            audit_writer.writeheader()

        for run_index, (comparison_id, image_a, image_b) in enumerate(remaining, 1):
            pair_index = int(comparison_id.split("-")[-1])
            seed_offset = pair_index * 10_000
            evaluations_ab, responses_ab = infer_mec(
                model, processor, files[image_a], files[image_b], args, seed_offset, timings,
            )
            evaluations_ba, responses_ba = infer_mec(
                model, processor, files[image_b], files[image_a], args, seed_offset + 5_000, timings,
            )
            calibrated = calibrate_scores(
                image_a, image_b, evaluations_ab, evaluations_ba, args.mec_samples,
                args.score_direction,
            )
            row: dict[str, object] = {
                "comparison_id": comparison_id, "image_a": image_a, "image_b": image_b,
                **calibrated,
                "model": args.model, "prompt_version": args.prompt_version,
                "prompt_sha256": args.prompt_sha256,
                "label_layout": LABEL_LAYOUT, "max_pixels": args.max_pixels, "quantization": QUANTIZATION,
                "cpu_offload": str(args.cpu_offload).lower(),
                "decode_mode": inference_decode_mode(args),
                "mec_samples": args.mec_samples, "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "raw_responses_ab": json.dumps(responses_ab, ensure_ascii=False),
                "raw_responses_ba": json.dumps(responses_ba, ensure_ascii=False),
            }
            audit_writer.writerow(row)
            audit_handle.flush()
            if row["status"] == "calibrated":
                write_vote(vote_writer, row)
                vote_handle.flush()
                accepted += 1

            completed_count = len(completed) + run_index
            if args.progress_every and (
                completed_count % args.progress_every == 0 or run_index == len(remaining)
            ):
                os.fsync(audit_handle.fileno())
                os.fsync(vote_handle.fileno())
                inferences = 2 * args.mec_samples * completed_count
                progress = (
                    f"Completed {completed_count}/{len(indexed_pairs)} unordered pairs "
                    f"({inferences} MEC directional samples); calibrated votes={accepted}"
                )
                print(progress, flush=True)
                if args.progress_file:
                    with args.progress_file.open("a", encoding="utf-8") as handle:
                        handle.write(progress + "\n")

    final_rows = load_audit(args.audit_output)
    accepted = rebuild_votes(final_rows, args.output)
    print(f"Completed {len(final_rows)} unordered pairs and "
          f"{2 * args.mec_samples * len(final_rows)} MEC directional samples.")
    print(f"Calibrated votes: {accepted}; excluded/invalid/tied pairs: {len(final_rows) - accepted}")
    print(f"Audit: {args.audit_output}")
    print(f"Votes: {args.output}")
    peak_cuda_gib = 0.0
    if torch.cuda.is_available():
        peak_cuda_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak CUDA memory allocated: {peak_cuda_gib:.2f} GiB")
    if args.timing_output:
        args.timing_output.parent.mkdir(parents=True, exist_ok=True)
        timing_report = {
            "model": args.model,
            "decode_mode": inference_decode_mode(args),
            "mec_samples": args.mec_samples,
            "mec_batch_size": args.mec_batch_size,
            "score_direction": args.score_direction,
            "completed_pairs": len(final_rows),
            "model_load_seconds": timings["model_load_seconds"],
            "preprocess_seconds": timings["preprocess_seconds"],
            "generate_seconds": timings["generate_seconds"],
            "generation_batches": int(timings["generation_batches"]),
            "wall_seconds": time.perf_counter() - run_started,
            "peak_cuda_allocated_gib": peak_cuda_gib,
        }
        args.timing_output.write_text(
            json.dumps(timing_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Timing: {args.timing_output}")


if __name__ == "__main__":
    main()
