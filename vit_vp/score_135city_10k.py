#!/usr/bin/env python3
"""Score the 135-city cohort with the 10k V8 ensemble; resume via SQLite."""
import argparse
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from extract_features import checkpoint_sha256, require_cuda
from score_oecd_50city_images_v8 import load_model_seed, preprocess_cpu, score_group

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/media/colin/白/GUVP_archive/oecd_europe_50city_10000')
MODEL = ROOT / 'output/vit_vp_10k_v8/deployment/finetuned_last2_v8'
CHECKPOINT = ROOT / 'models/checkpoints/dinov3_vitb.safetensors'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.batch_size <= 16 or args.limit < 0:
        raise ValueError('Invalid batch size or limit')
    manifest_path = ARCHIVE / 'images_manifest_expanded_1000.csv'
    manifest = pd.read_csv(manifest_path, dtype={'uuid': str, 'orig_id': str, 'oecd_region_code': str})
    if len(manifest) != 135000 or manifest.uuid.nunique() != 135000 or len(manifest.groupby('city')) != 135 or not manifest.groupby('city').size().eq(1000).all():
        raise ValueError('Expected 135 unique cities and 1,000 unique images per city')
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    metadata = json.loads((MODEL / 'model.json').read_text())
    if metadata['training_images'] != 10000 or set(metadata['seed_ensemble']) != {42, 137, 2026} or metadata['feature_cache_edge'] != 512:
        raise ValueError('Unexpected model')
    if checkpoint_sha256(CHECKPOINT) != metadata['source_checkpoint_sha256']:
        raise ValueError('Base checkpoint hash mismatch')
    device = require_cuda(torch.device('cuda'))
    models = []
    for artifact in metadata['artifacts']:
        weight = MODEL / artifact['weights']
        if checkpoint_sha256(weight) != artifact['sha256']:
            raise ValueError(f'Weight hash mismatch: {weight}')
        encoder, head = load_model_seed(CHECKPOINT, weight, device)
        models.append((artifact, encoder, head))
    run_info = {'manifest_sha256': checkpoint_sha256(manifest_path), 'model_json_sha256': checkpoint_sha256(MODEL / 'model.json'),
                'image_count': len(manifest), 'limit': args.limit, 'model': '10k V8 3-seed ensemble',
                'weights': {str(a['seed']): a['sha256'] for a in metadata['artifacts']},
                'checkpoint_sha256': metadata['source_checkpoint_sha256'], 'edge': 512,
                'training_overlap': '10,000 images are part of the 135,000 image cohort; their predictions are in-sample',
                'labels': 'V8 Qwen/TrueSkill relative labels; not human ground truth'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info_path = args.output_dir / 'run_metadata.json'
    if info_path.exists() and json.loads(info_path.read_text()) != run_info:
        raise ValueError('Existing run metadata does not match; refusing to mix results')
    info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + '\n')
    db = sqlite3.connect(args.output_dir / 'predictions.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS predictions (idx INTEGER PRIMARY KEY, image_id TEXT UNIQUE NOT NULL, score REAL NOT NULL, seed42 REAL NOT NULL, seed137 REAL NOT NULL, seed2026 REAL NOT NULL)')
    done = {row[0] for row in db.execute('SELECT idx FROM predictions')}
    if done - set(range(len(manifest))):
        raise ValueError('Existing predictions outside manifest range')
    records = manifest.to_dict('records')
    pending = [(i, row) for i, row in enumerate(records) if i not in done]
    started = time.monotonic()
    for offset in range(0, len(pending), 64):
        groups = defaultdict(list)
        for idx, row in pending[offset:offset+64]:
            path = ARCHIVE / 'images' / f"{row['uuid']}.jpg"
            pixels, width, height = preprocess_cpu(path, 512)
            groups[(width, height)].append((idx, row, pixels))
        batch_results = []
        for group in groups.values():
            for start in range(0, len(group), args.batch_size):
                batch = group[start:start+args.batch_size]
                pred = score_group(torch.cat([entry[2] for entry in batch]), models, device)
                for j, (idx, row, _) in enumerate(batch):
                    vals = [float(pred[s][j]) for s in (42, 137, 2026)]
                    if not all(np.isfinite(vals)):
                        raise ValueError(f"Nonfinite prediction: {row['uuid']}")
                    batch_results.append((idx, row['uuid'], float(np.mean(vals)), *vals))
        with db:
            db.executemany('INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)', batch_results)
        completed = len(done) + min(offset + 64, len(pending))
        print(f'{completed}/{len(manifest)} images; elapsed={time.monotonic()-started:.1f}s; peak_cuda_gib={torch.cuda.max_memory_allocated()/1024**3:.3f}', flush=True)
    predictions = pd.read_sql_query('SELECT * FROM predictions ORDER BY idx', db)
    if len(predictions) != len(manifest) or predictions.image_id.tolist() != manifest.uuid.tolist():
        raise ValueError('Prediction coverage or order does not match manifest')
    result = manifest[['uuid', 'orig_id', 'city', 'country', 'oecd_region', 'oecd_region_code', 'lat', 'lon']].copy()
    result = result.rename(columns={'uuid': 'image_id', 'orig_id': 'mapillary_image_id'})
    result['predicted_vp_score'] = predictions.score
    for seed in (42, 137, 2026):
        result[f'seed_{seed}_score'] = predictions[f'seed{seed}']
    result['model_version'] = 'DINOv3 V8 10k 3-seed ensemble'
    target = args.output_dir / 'image_predictions.csv'
    temp = target.with_suffix('.csv.tmp')
    result.to_csv(temp, index=False)
    temp.replace(target)
    summary = result.groupby(['city', 'country'], as_index=False).agg(image_count=('image_id', 'size'), mean_vp=('predicted_vp_score', 'mean'), median_vp=('predicted_vp_score', 'median'), sd_vp=('predicted_vp_score', 'std'))
    summary.to_csv(args.output_dir / 'city_summary.csv', index=False)
    print(f'Complete: {len(result)} images -> {target}', flush=True)

if __name__ == '__main__':
    main()
