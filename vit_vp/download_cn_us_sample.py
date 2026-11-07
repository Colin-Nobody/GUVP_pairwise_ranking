#!/usr/bin/env python3
"""Download and audit a resumable, grid-balanced Mapillary sample per city."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sqlite3
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from download_oecd_50city_mapillary import download_one, load_token
from prepare_oecd_50city_mapillary_sample import grid_key, haversine_km
from survey_mapillary_cn_us import CITIES

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/media/colin/白/GUVP_archive')
OUT = ARCHIVE / 'mapillary_cn_us_202611'
SURVEY6 = ROOT / 'output/mapillary_cn_us_survey_202611'
SURVEY12 = ROOT / 'output/mapillary_cn_survey_12km'
OTHER12 = ROOT / 'output/mapillary_cn_other_survey_12km'
SEED = 20261107

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def ordered_candidates(city, lat, lon):
    sources = [(SURVEY6, 6), (SURVEY12, 12), (OTHER12, 12)]
    by_seq = {}
    for folder, radius in sources:
        p = folder / (city.replace(' ', '_') + '_candidates.csv')
        if not p.is_file():
            continue
        with p.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                row['lat'], row['lon'] = float(row['lat']), float(row['lon'])
                row['distance_km'] = haversine_km(lat, lon, row['lat'], row['lon'])
                row['survey_radius'] = radius
                seq = row['sequence_id']
                if seq not in by_seq or row['distance_km'] < by_seq[seq]['distance_km']:
                    by_seq[seq] = row
    # Prioritize the same expanding-radius policy as the European sample.
    rng = random.Random(SEED + sum(map(ord, city)))
    output = []
    for lower, upper in [(0,6),(6,8),(8,10),(10,12)]:
        buckets = defaultdict(list)
        for row in by_seq.values():
            if lower < row['distance_km'] <= upper or (lower == 0 and row['distance_km'] == 0):
                buckets[grid_key(row['lat'], row['lon'])].append(row)
        keys = sorted(buckets); rng.shuffle(keys)
        for key in keys:
            buckets[key].sort(key=lambda x: x['image_id']); rng.shuffle(buckets[key])
        while keys:
            remaining = []
            for key in keys:
                if buckets[key]: output.append(buckets[key].pop())
                if buckets[key]: remaining.append(key)
            keys = remaining
    return output

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cities', nargs='+', required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    roster = {x[0]:x for x in CITIES}
    if any(city not in roster or city == 'Beijing' for city in args.cities):
        raise ValueError('Unknown or explicitly excluded city')
    OUT.mkdir(parents=True, exist_ok=True)
    token, _ = load_token()
    db = sqlite3.connect(OUT / 'download_audit.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS attempts (image_id TEXT PRIMARY KEY, city TEXT, sequence_id TEXT, status TEXT, sha256 TEXT, bytes INTEGER, detail TEXT, uuid TEXT, lat REAL, lon REAL, distance_km REAL, quality_score TEXT, survey_radius INTEGER, sample_order INTEGER)')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS unique_success_hash ON attempts(sha256) WHERE status="downloaded"')
    db.commit()
    existing_hashes = set()
    integrity = ARCHIVE / 'oecd_europe_50city_10000/image_integrity_expanded_1000.csv'
    with integrity.open(encoding='utf-8-sig', newline='') as f:
        existing_hashes.update(row['sha256'] for row in csv.DictReader(f) if row.get('sha256'))
    print('Existing European image hashes loaded:', len(existing_hashes), flush=True)
    for city in args.cities:
        _, country, center_lat, center_lon = roster[city]
        ordered = ordered_candidates(city, center_lat, center_lon)
        if not ordered:
            print(f'{city}: no candidate file available', flush=True); continue
        folder = OUT / 'images' / city.replace(' ', '_'); folder.mkdir(parents=True, exist_ok=True)
        done = {x[0] for x in db.execute('SELECT image_id FROM attempts')}
        successes = db.execute('SELECT count(*) FROM attempts WHERE city=? AND status="downloaded"',(city,)).fetchone()[0]
        print(f'{city}: {len(ordered)} candidate sequences; existing valid={successes}', flush=True)
        while successes < 1000:
            if shutil.disk_usage(OUT).free < 1024**3:
                raise RuntimeError('Stopped: archive disk has less than 1 GiB free')
            pending = [(i,r) for i,r in enumerate(ordered) if r['image_id'] not in done][:min(24,1000-successes)]
            if not pending: break
            tasks=[]
            for index,row in pending:
                stable_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL,'guvp-mapillary-cn-us-'+row['image_id']))
                tasks.append((index,row,{'uuid':stable_uuid,'city':city,'orig_id':row['image_id']}))
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results=list(executor.map(lambda x: download_one(x[2],folder,token),tasks))
            for (index,row,task),result in zip(tasks,results):
                image_id=row['image_id']; path=folder/(task['uuid']+'.jpg'); status=result['status']; sha=None
                if status in ('downloaded','skipped_existing'):
                    sha=digest(path)
                    duplicate=sha in existing_hashes or db.execute('SELECT 1 FROM attempts WHERE sha256=? AND status="downloaded"',(sha,)).fetchone() is not None
                    if duplicate:
                        status='duplicate_content'; path.unlink(missing_ok=True)
                    else:
                        status='downloaded'; successes+=1
                db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (image_id,city,row['sequence_id'],status,sha,int(result.get('bytes',0)),result.get('detail',''),task['uuid'],row['lat'],row['lon'],row['distance_km'],row['quality_score'],row['survey_radius'],index))
                db.commit(); done.add(image_id)
            print(f'{city}: {successes}/1000 verified; {len(done)} attempted overall; free_gib={shutil.disk_usage(OUT).free/1024**3:.2f}',flush=True)
        print(f'{city}: finished with {successes}/1000 verified images',flush=True)
    columns=[row[1] for row in db.execute('PRAGMA table_info(attempts)')]
    with (OUT/'download_audit.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(columns); w.writerows(db.execute('SELECT * FROM attempts ORDER BY city,sample_order'))
    stats={city:db.execute('SELECT status,count(*) FROM attempts WHERE city=? GROUP BY status',(city,)).fetchall() for city in args.cities}
    (OUT/'latest_summary.json').write_text(json.dumps({'cities':stats,'target_per_city':1000,'seed':SEED,'candidate_rules':'6km first, then 8/10/12km, one image per sequence, grid round-robin','training':'No model inference or training performed'},ensure_ascii=False,indent=2)+'\n')
    print('Download audit:', json.dumps(stats,ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
