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

BRANCH = "qwen-5k-results"
DEST = "experiments/qwen-5k"
RUNS = {
    "qwen25": "output/vlm_q25_5k_v8_pairbatch50_5day_mec3",
    "qwen3": "output/vlm_qwen3_5k_v8_pairbatch50_mec3",
}
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

def launch_code():
    roots = [path for path in Path(".").glob("hpc*") if path.is_dir()]
    expected = {"q25.sh", "q25.sbatch", "q5.sh", "q5.sbatch"}
    available = {p.name: p for root in roots for p in root.glob("*") if p.is_file()}
    if not expected <= set(available):
        raise ValueError("Missing launchers: " + str(expected - set(available)))
    paths = {available[name] for name in expected}
    pending = list(paths)
    while pending:
        launcher = pending.pop()
        for name in re.findall(r"[\w.-]+\.(?:py|sh|sbatch)", launcher.read_text()):
            local = launcher.parent / name
            root = Path(name)
            path = next((p for p in (local, root, available.get(name)) if p and p.is_file()), None)
            if path and path not in paths:
                paths.add(path)
                if path.suffix in {".sh", ".sbatch"}:
                    pending.append(path)
    for name in ("vlm_pairwise.py", "prepare_global_streetscapes_sample.py"):
        if Path(name).is_file():
            paths.add(Path(name))
    return sorted(paths)

def main():
    base = git("rev-parse", "FETCH_HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git("read-tree", base, env=env)
        manifest = {}
        for model, folder in RUNS.items():
            source = Path(folder)
            prefix = f"{DEST}/{model}"
            manifest[model] = {}
            for name in FILES:
                payload = (source / name).read_bytes()
                if name == "perception_scores.csv" and len(list(csv.DictReader(io.StringIO(payload.decode())))) != 5000:
                    raise ValueError("Unexpected image count: " + model)
                add(f"{prefix}/{name}", payload, env)
            for name in ("pair_audit.csv", "vlm_pairwise_votes.csv"):
                payload, count = sanitize_csv(source / name)
                if name == "pair_audit.csv" and count != 32500:
                    raise ValueError("Unexpected pair count: " + model)
                add(f"{prefix}/{name}.gz", gzip.compress(payload, mtime=0), env)
                manifest[model][name] = count
        code = launch_code()
        for path in code:
            payload = path.read_bytes()
            if re.search(rb"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|https?://[^/\s]+:[^/\s]+@github\.com", payload):
                raise ValueError("Review credentials in " + str(path))
            add(f"{DEST}/code/{path.as_posix()}", payload, env)
        manifest["code_paths"] = [p.as_posix() for p in code]
        manifest["jobs"] = {"qwen25": 391888, "qwen3": 391832}
        manifest["hpc_checkout_commit"] = git("rev-parse", "HEAD")
        add(f"{DEST}/manifest.json", (json.dumps(manifest, indent=2) + "\n").encode(), env)
        tree = git("write-tree", env=env)
        commit = git("commit-tree", tree, "-p", base, "-m", "Archive both Qwen V8 5K experiment results and code")
        subprocess.run(["git", "push", "origin", f"{commit}:refs/heads/{BRANCH}"], check=True)
        print(commit)
        print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
