#!/usr/bin/env python3
"""Survey live Mapillary non-panoramic candidates around selected city centers."""
import csv
import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from prepare_oecd_50city_mapillary_sample import (TILE_URL, ZOOM, decode_image_layer,
    grid_key, haversine_km, read_local_token, tile_for)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/mapillary_cn_us_survey_202611'
RADIUS = 6.0
CITIES = [
 ('Beijing','China',39.9042,116.4074), ('Shanghai','China',31.2304,121.4737),
 ('Guangzhou','China',23.1291,113.2644), ('Shenzhen','China',22.5431,114.0579),
 ('Chengdu','China',30.5728,104.0668), ('Wuhan','China',30.5928,114.3055),
 ('Hangzhou','China',30.2741,120.1551), ('Nanjing','China',32.0603,118.7969),
 ('New York','United States',40.7128,-74.0060), ('Los Angeles','United States',34.0522,-118.2437),
 ('Chicago','United States',41.8781,-87.6298), ('San Francisco','United States',37.7749,-122.4194),
 ('Boston','United States',42.3601,-71.0589), ('Seattle','United States',47.6062,-122.3321),
 ('Washington DC','United States',38.9072,-77.0369), ('Philadelphia','United States',39.9526,-75.1652),
]

def city_tiles(lat, lon):
    dlat = RADIUS / 110.574
    dlon = RADIUS / (111.320 * math.cos(math.radians(lat)))
    xmin, ymax = tile_for(lon-dlon, lat-dlat, ZOOM)
    xmax, ymin = tile_for(lon+dlon, lat+dlat, ZOOM)
    return [(x,y) for x in range(xmin,xmax+1) for y in range(ymin,ymax+1)]

def main():
    global RADIUS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=float, default=6.0)
    parser.add_argument('--cities', nargs='*')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    if args.radius <= 0 or args.radius > 12:
        raise ValueError('Radius must be in (0, 12] km')
    RADIUS = args.radius
    OUT = args.output_dir
    cities = [item for item in CITIES if not args.cities or item[0] in args.cities]
    if args.cities and len(cities) != len(set(args.cities)):
        raise ValueError('Unknown or duplicate city')
    OUT.mkdir(parents=True, exist_ok=True)
    token = read_local_token()
    completed = []
    for name, country, lat, lon in cities:
        candidates = {}; failures = []; tiles = city_tiles(lat, lon)
        def fetch(tile):
            x,y = tile
            for attempt in range(4):
                try:
                    r = requests.get(TILE_URL.format(z=ZOOM,x=x,y=y), params={'access_token':token}, timeout=(10,50))
                    if r.status_code == 200:
                        return tile, decode_image_layer(r.content,x,y,ZOOM), None
                    if r.status_code not in (429,500,502,503,504):
                        return tile, [], f'HTTP {r.status_code}'
                except (requests.RequestException, ValueError, IndexError, KeyError):
                    pass
                time.sleep(2**attempt)
            return tile, [], 'network, parsing or retry failure'
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(fetch,t) for t in tiles]):
                tile, rows, error = future.result()
                if error:
                    failures.append({'tile':tile, 'error':error}); continue
                for row in rows:
                    try:
                        seq=str(row.get('sequence_id') or '').strip(); img=row.get('id') or row.get('image_id')
                        lat2=float(row['lat']); lon2=float(row['lon']); quality=row.get('quality_score')
                        if not seq or img is None or haversine_km(lat,lon,lat2,lon2)>RADIUS or (quality is not None and float(quality)<0.25):
                            continue
                        old=candidates.get(seq)
                        if old is None or float(quality or 0)>float(old['quality_score'] or 0):
                            candidates[seq]={'image_id':str(img),'sequence_id':seq,'lat':lat2,'lon':lon2,'quality_score':quality}
                    except (TypeError,ValueError):
                        continue
        rows=list(candidates.values())
        result={'city':name,'country':country,'center_lat':lat,'center_lon':lon,'radius_km':RADIUS,
                'tiles_total':len(tiles),'tiles_failed':len(failures),'unique_sequences':len(rows),
                'occupied_grid_cells':len({grid_key(r['lat'],r['lon']) for r in rows}),
                'has_1000_candidates':len(rows)>=1000,'complete_tile_query':not failures,
                'quality_missing':sum(r['quality_score'] is None for r in rows)}
        completed.append(result)
        with (OUT / f'{name.replace(" ","_")}_candidates.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['image_id','sequence_id','lat','lon','quality_score']); writer.writeheader(); writer.writerows(rows)
        with (OUT/'city_candidate_counts.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(result)); writer.writeheader(); writer.writerows(completed)
        (OUT/'latest_audit.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'completed':completed,'latest_failures':failures,
            'rules':f'Mapillary live vector tiles, non-panoramic, within {RADIUS:g} km of supplied center, one image per sequence, quality >=0.25 when available; 0.01-degree spatial grid proxy',
            'limitations':'Candidates only; individual files have not been downloaded, decoded or content-deduplicated. Areas and centers are approximate; comparisons use metadata snapshot at query time.'},ensure_ascii=False,indent=2)+'\n')
        print(f'{name}: {len(rows)} sequences, {result["occupied_grid_cells"]} grid cells, {len(failures)}/{len(tiles)} tiles failed',flush=True)

if __name__=='__main__':
    main()
