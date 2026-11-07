#!/usr/bin/env python3
"""Compare city ranks from the 1,200-image and 10,000-image V8 ensembles."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/media/colin/白/GUVP_archive/oecd_europe_50city_10000/image_vp_predictions_expanded_1000.csv')
NEW = ROOT / 'output/vit_vp_10k_v8/score_135city/image_predictions.csv'
OUT = ROOT / 'output/vit_vp_10k_v8/compare_135city_models'
OUT.mkdir(parents=True, exist_ok=True)
old = pd.read_csv(OLD, usecols=['image_id', 'city', 'predicted_vp_score', 'model_version'])
new = pd.read_csv(NEW, usecols=['image_id', 'city', 'country', 'predicted_vp_score', 'model_version'])
assert len(old) == len(new) == 135000 and old.image_id.is_unique and new.image_id.is_unique
assert old.model_version.eq('DINOv3 ViT-B/16 V8 all-1200 fit; 3-seed ensemble').all()
assert new.model_version.eq('DINOv3 V8 10k 3-seed ensemble').all()
paired = old.merge(new, on='image_id', validate='one_to_one', suffixes=('_old', '_new'))
assert len(paired) == 135000 and paired.city_old.eq(paired.city_new).all()
assert np.isfinite(paired[['predicted_vp_score_old', 'predicted_vp_score_new']].to_numpy()).all()
cities = paired.groupby(['city_new', 'country'], as_index=False).agg(
    images=('image_id', 'size'), old_mean=('predicted_vp_score_old', 'mean'), new_mean=('predicted_vp_score_new', 'mean'))
assert len(cities) == 135 and cities.images.eq(1000).all()
cities = cities.rename(columns={'city_new': 'city'})
for label in ('old', 'new'):
    cities[f'{label}_rank'] = cities[f'{label}_mean'].rank(method='min', ascending=False).astype(int)
cities['rank_change_higher'] = cities.old_rank - cities.new_rank
cities['new_minus_old_mean'] = cities.new_mean - cities.old_mean
cities = cities.sort_values(['new_rank', 'city'])
cities.to_csv(OUT / 'city_rankings_135.csv', index=False, float_format='%.8f')
rho = float(spearmanr(cities.old_rank, cities.new_rank).statistic)
info = {'city_count': 135, 'images_per_city': 1000, 'paired_images': len(paired), 'city_rank_spearman': rho,
        'rank_displacement_median': float(cities.rank_change_higher.abs().median()),
        'rank_displacement_max': int(cities.rank_change_higher.abs().max()),
        'rank_direction': '1 = highest predicted visual pollution; positive rank_change_higher means higher in 10k model',
        'old': str(OLD), 'new': str(NEW),
        'limitations': 'Identical images but models and independently normalized VLM training labels differ; 10000 new-model training images are included. Descriptive model comparison, not ground truth or change over time.'}
(OUT / 'comparison_metadata.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
ordered = cities.sort_values('old_rank', ascending=False)
fig, ax = plt.subplots(figsize=(13, 36), layout='constrained')
y = np.arange(len(ordered))
for pos, row in enumerate(ordered.itertuples()):
    ax.plot([row.old_rank, row.new_rank], [pos, pos], color='#bdc6d2', lw=1.2, zorder=1)
ax.scatter(ordered.old_rank, y, color='#4785bd', s=22, label='Old: trained on 1,200 images', zorder=2)
ax.scatter(ordered.new_rank, y, color='#df733f', s=22, label='New: trained on 10,000 images', zorder=3)
ax.set_yticks(y, ordered.city, fontsize=6.7)
ax.invert_xaxis(); ax.set_xlim(138, -2); ax.set_xlabel('City rank (1 = highest predicted visual pollution)')
ax.set_title(f'135 cities: old vs new model rankings on the same 135,000 images\nSpearman rank correlation = {rho:.3f}; each city has 1,000 images', loc='left', fontsize=13, weight='bold')
ax.grid(axis='x', alpha=.2); ax.legend(loc='lower center', ncol=2, frameon=False)
fig.savefig(OUT / 'rank_comparison_135.png', dpi=170); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout='constrained')
ax = axes[0]; ax.scatter(cities.old_rank, cities.new_rank, c='#367e9b', alpha=.75, s=32)
ax.plot([1, 135], [1, 135], color='#9a9a9a', linestyle='--', lw=1)
ax.invert_xaxis(); ax.invert_yaxis(); ax.set_xlabel('Old model rank'); ax.set_ylabel('New model rank')
ax.set_title(f'Rank agreement: Spearman {rho:.3f}', loc='left', weight='bold'); ax.grid(alpha=.16)
ax = axes[1]; top = cities.assign(magnitude=cities.rank_change_higher.abs()).nlargest(15, 'magnitude').sort_values('rank_change_higher')
ax.barh(top.city, top.rank_change_higher, color=['#4785bd' if v < 0 else '#df733f' for v in top.rank_change_higher])
ax.axvline(0, color='#666', lw=.8); ax.set_xlabel('Rank places moved up in new model')
ax.set_title('15 largest changes in city rank', loc='left', weight='bold'); ax.grid(axis='x', alpha=.16)
fig.suptitle('135-city comparison: 1,200-image vs 10,000-image training', fontsize=15, weight='bold')
fig.savefig(OUT / 'rank_comparison_summary.png', dpi=180); plt.close(fig)
print(json.dumps(info, indent=2))
print('Highest predicted pollution:', cities.head(5)[['city','new_mean','new_rank']].to_string(index=False))
print('Lowest predicted pollution:', cities.tail(5)[['city','new_mean','new_rank']].to_string(index=False))
