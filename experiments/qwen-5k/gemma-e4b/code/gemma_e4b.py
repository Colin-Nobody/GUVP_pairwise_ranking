"""Use a local llama-server with the original GUVP V8 audit and calibration."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from io import BytesIO
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

import vlm_pairwise as pipeline


def image_part(path):
    try:
        raw = path.read_bytes()
        with Image.open(BytesIO(raw)) as img:
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="PNG")
            raw = buffer.getvalue()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode street-view image {path}: {exc}") from exc
    data = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def complete(images, paths, args, seed):
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Image A:"}, images[0],
            {"type": "text", "text": "Image B:"}, images[1],
            {"type": "text", "text": args.prompt},
        ]}],
        "seed": seed,
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
    }
    request = Request(
        "http://127.0.0.1:" + os.environ.get("GEMMA_PORT", "18080")
        + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
            choice = result["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
    except HTTPError as exc:
        detail = exc.read(512).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemma HTTP {exc.code} for {paths[0]} and {paths[1]}: {detail}"
        ) from exc
    if not isinstance(content, str):
        raise ValueError("Gemma did not return text content")
    if choice.get("finish_reason") == "length" or pipeline.parse_evaluation(content) is None:
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        tokens = (result.get("usage") or {}).get("completion_tokens")
        print(
            f"Gemma response for {paths[0].name} and {paths[1].name}: "
            f"finish_reason={choice.get('finish_reason')!r}, completion_tokens={tokens}, "
            f"content_chars={len(content)}, reasoning_chars={len(reasoning)}",
            flush=True,
        )
    return content


def infer_mec(model, processor, image_a, image_b, args, seed_offset, timings=None):
    started = time.perf_counter()
    images = [image_part(image_a), image_part(image_b)]
    if timings is not None:
        timings["preprocess_seconds"] += time.perf_counter() - started
    started = time.perf_counter()
    responses = [
        complete(images, (image_a, image_b), args, args.seed + seed_offset + i)
        for i in range(args.mec_samples)
    ]
    if timings is not None:
        timings["generate_seconds"] += time.perf_counter() - started
        timings["generation_batches"] += args.mec_samples
    scores = [score for text in responses if (score := pipeline.parse_evaluation(text)) is not None]
    return scores, responses


def infer_pair_batch(model, processor, pairs, args, seed_offsets, timings=None):
    started = time.perf_counter()
    images = [[image_part(a), image_part(b)] for a, b in pairs]
    if timings is not None:
        timings["preprocess_seconds"] += time.perf_counter() - started
    responses = [[] for _ in pairs]
    with ThreadPoolExecutor(max_workers=len(pairs)) as workers:
        for sample in range(args.mec_samples):
            started = time.perf_counter()
            jobs = [
                workers.submit(complete, encoded, paths, args, args.seed + offset + sample)
                for encoded, paths, offset in zip(images, pairs, seed_offsets, strict=True)
            ]
            for history, job in zip(responses, jobs, strict=True):
                history.append(job.result())
            if timings is not None:
                timings["generate_seconds"] += time.perf_counter() - started
                timings["generation_batches"] += 1
    scores = [
        [score for text in history if (score := pipeline.parse_evaluation(text)) is not None]
        for history in responses
    ]
    return scores, responses


if __name__ == "__main__":
    if not hasattr(pipeline, "infer_mec_pair_batch"):
        raise RuntimeError("The HPC batch-enabled vlm_pairwise.py is required")
    pipeline.SUPPORTED_MODELS.add("google/gemma-4-E4B-it-qat-q4_0-gguf")
    pipeline.QUANTIZATION = "GGUF Q4_0"
    pipeline.load_model = lambda args: (None, None)
    pipeline.infer_mec = infer_mec
    pipeline.infer_mec_pair_batch = infer_pair_batch
    pipeline.main()
