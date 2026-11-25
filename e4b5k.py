"""Export the two completed V8 Qwen 5K runs from the HPC checkout."""
import csv
import gzip
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

RAW_FIELDS = {"raw_responses_ab", "raw_responses_ba", "raw_response", "raw_responses", "prompt"}
FILES = ("perception_scores.csv", "ranking_summary.json", "timing.json")

def git(*args, **kwargs):
    return subprocess.check_output(["git", *args], **kwargs).decode().strip()

def sanitize_csv(path):
    data = io.StringIO(newline="")
    with path.open(newline="", encoding="utf-8") as handle:
        source = csv.DictReader(handle)
        fields = [field for field in source.fieldnames if field.lower() not in RAW_FIELDS
                  and not field.lower().startswith(("raw_", "response", "thinking", "reasoning"))
                  and field.lower() not in {"prompt_text", "prompt_body"}]
        writer = csv.DictWriter(data, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        count = 0
        for row in source:
            writer.writerow(row)
            count += 1
    return data.getvalue().encode(), count

def add(path, payload, env):
    if len(payload) >= 100 * 1024**2:
        raise ValueError("Oversized Git file: " + path)
    blob = git("hash-object", "-w", "--stdin", input=payload)
    git("update-index", "--add", "--cacheinfo", "100644", blob, path, env=env)


BRANCH = "gemma-e4b-5k-results"
DEST = "experiments/qwen-5k/gemma-e4b"
SOURCE = Path("output/e4b_v8_5k_391681/batch_7")

def rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))

def pairs(records):
    return {tuple(sorted(Path(row[key]).name for key in ("image_a", "image_b"))) for row in records}

def rank(votes, target):
    candidates = ["python3", "python"]
    for root in (Path("."), Path.home()):
        for pattern in ("*/bin/python", "*/envs/*/bin/python", ".*/bin/python", ".*/envs/*/bin/python"):
            candidates.extend(str(path.resolve().parent / path.name) if not path.is_symlink()
                              else str(path.absolute()) for path in root.glob(pattern))
    for executable in dict.fromkeys(candidates):
        try:
            check = subprocess.run([executable, "-c", "import pandas, matplotlib, trueskill"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            continue
        if check.returncode == 0:
            subprocess.run([executable, "main.py", "--input", str(votes), "--output-dir", str(target),
                            "--dimension", "visual_pollution_v8"], check=True)
            return target
    print("Ranking dependencies unavailable on this node. Archiving verified pair results.", flush=True)
    return None

def main():
    base = git("rev-parse", "refs/remotes/origin/" + BRANCH)
    audit, n_audit = sanitize_csv(SOURCE / "audit.csv")
    votes, n_votes = sanitize_csv(SOURCE / "votes.csv")
    if (n_audit, n_votes) != (32500, 32125):
        raise ValueError("Unexpected audit/vote counts: " + str((n_audit, n_votes)))
    gemma_pairs = pairs(rows(audit))
    qwen_audit = subprocess.check_output(["git", "show", base + ":experiments/qwen-5k/qwen25/pair_audit.csv.gz"])
    if len(gemma_pairs) != n_audit or gemma_pairs != pairs(rows(gzip.decompress(qwen_audit))):
        raise ValueError("Gemma and Qwen do not share the same unique pair set")
    image_ids = {image for pair in gemma_pairs for image in pair}
    if len(image_ids) != 5000:
        raise ValueError("Expected 5,000 images")
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        env = dict(os.environ, GIT_INDEX_FILE=str(tmp / "index"))
        git("read-tree", base, env=env)
        for name, payload in (("pair_audit.csv.gz", gzip.compress(audit)),
                              ("vlm_pairwise_votes.csv.gz", gzip.compress(votes)),
                              ("timing.json", (SOURCE / "timing.json").read_bytes())):
            add(DEST + "/" + name, payload, env)
        ranking = next((directory for directory in (SOURCE, SOURCE.parent)
                        if (directory / "perception_scores.csv").is_file()), None)
        if ranking is None:
            clean_votes = tmp / "votes.csv"
            clean_votes.write_bytes(votes)
            ranking = rank(clean_votes, tmp / "ranking")
        if ranking:
            score_payload = (ranking / "perception_scores.csv").read_bytes()
            score_rows = rows(score_payload)
            if len(score_rows) != 5000 or {Path(row["image_id"]).name for row in score_rows} != image_ids:
                raise ValueError("Ranked image IDs do not match the audit")
            add(DEST + "/perception_scores.csv", score_payload, env)
            if (ranking / "ranking_summary.json").is_file():
                add(DEST + "/ranking_summary.json", (ranking / "ranking_summary.json").read_bytes(), env)
        launcher = Path("e4b5k.sbatch")
        code = {launcher, Path("main.py"), Path("vlm_pairwise.py")}
        for item in re.findall(r"[\w./-]+\.(?:py|sh|sbatch)", launcher.read_text()):
            path = Path(item)
            if not path.is_absolute() and path.is_file():
                code.add(path)
        for item in sorted(code):
            payload = item.read_bytes()
            if re.search(rb"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|https?://[^/\s]+:[^/\s]+@github\.com", payload):
                raise ValueError("Review credentials in " + str(item))
            add(DEST + "/code/" + item.as_posix(), payload, env)
        manifest = {"job": 391681, "pair_batch_size": 7, "unordered_pairs": n_audit,
                    "valid_votes": n_votes, "excluded_invalid_or_tied": n_audit - n_votes,
                    "directional_samples": 195000, "images": len(image_ids), "pairs_match_qwen": True,
                    "ranking_included": bool(ranking), "code_paths": sorted(str(path) for path in code)}
        add(DEST + "/manifest.json", (json.dumps(manifest, indent=2) + "\n").encode(), env)
        tree = git("write-tree", env=env)
        commit = git("-c", "user.name=Codex", "-c", "user.email=codex@users.noreply.github.com",
                     "commit-tree", tree, "-p", base, "-m", "Archive Gemma E4B V8 5K results")
        subprocess.run(["git", "update-ref", "refs/heads/gemma-e4b-5k-export", commit], check=True)
        print("Created commit " + commit, flush=True)
        print(json.dumps(manifest), flush=True)
        subprocess.run(["git", "push", "origin", commit + ":refs/heads/" + BRANCH], check=True)

if __name__ == "__main__":
    main()
