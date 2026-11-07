#!/usr/bin/env python3
"""Reconcile the HPC 10k selection and labels with the local image archive."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/media/colin/白/GUVP_archive/oecd_europe_50city_10000')
OUT = ROOT / 'output/vit_vp_10k_v8'
manifest_path = ROOT / 'data/vlm_hpc_10k/manifest_10000.csv'
score_path = ROOT / 'output/vlm_hpc_10k_v8_pairbatch50_5day_mec3/perception_scores.csv'
selected_path = OUT / 'finetune_last2/inner_selected_configs.csv'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

manifest = pd.read_csv(manifest_path, dtype={'uuid': str, 'sha256': str})
scores = pd.read_csv(score_path)
scores['uuid'] = scores.image_id.str.removesuffix('.jpg')
assert len(manifest) == len(scores) == 10000
assert manifest.uuid.is_unique and scores.uuid.is_unique
assert set(manifest.uuid) == set(scores.uuid)
assert (manifest.image_filename == manifest.uuid + '.jpg').all()
assert (manifest.decode == 'ok').all()
selected = pd.read_csv(selected_path)
assert len(selected) == 15 and selected.groupby('repeat').size().eq(5).all()
merged = manifest.merge(scores[['uuid', 'dimension', 'visual_pollution_score_0_10']], on='uuid', validate='one_to_one')
assert merged.dimension.eq('visual_pollution_v8').all()
assert merged.visual_pollution_score_0_10.between(0, 10).all()
paths = []
for n, row in enumerate(merged.itertuples(index=False), 1):
    path = ARCHIVE / 'images' / row.image_filename
    if sha256(path) != row.sha256:
        raise ValueError(f'Image checksum mismatch: {path}')
    paths.append(str(path))
    if n % 1000 == 0:
        print(f'Image SHA-256 verified: {n}/10000', flush=True)
out = pd.DataFrame({'image_id': merged.uuid, 'image_path': paths, 'city': merged.city,
                    'target_visual_pollution_score_0_10': merged.visual_pollution_score_0_10})
OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / 'image_manifest.csv', index=False)
provenance = {'label': 'V8 Qwen/TrueSkill visual_pollution_score_0_10; VLM-derived, not human ground truth',
              'result_commit': '5843efc94274a05c342b54c515543b7fbe518b6e',
              'source_manifest': str(manifest_path), 'source_manifest_sha256': sha256(manifest_path),
              'scores': str(score_path), 'scores_sha256': sha256(score_path),
              'selected_configs': str(selected_path), 'selected_configs_sha256': sha256(selected_path),
              'image_archive': str(ARCHIVE), 'image_sha256_verified': len(out)}
(OUT / 'label_provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')
print('10k manifest and provenance ready', flush=True)
