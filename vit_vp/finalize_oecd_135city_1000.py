#!/usr/bin/env python3
"""Promote exactly 300 valid, content-unique supplements for each of 135 OECD cities."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify(path: Path) -> tuple[int,int]:
    with Image.open(path) as im:
        if im.format!='JPEG': raise ValueError(f'not JPEG: {path}')
        im.verify()
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB'); im.load(); return im.size


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--supplement-dir',type=Path,required=True)
    p.add_argument('--initial-manifest',type=Path,required=True)
    p.add_argument('--initial-integrity',type=Path,required=True)
    p.add_argument('--replacement-manifests',nargs='+',type=Path,required=True)
    p.add_argument('--output-manifest',type=Path,required=True)
    p.add_argument('--output-integrity',type=Path,required=True)
    p.add_argument('--per-city',type=int,default=1000)
    a=p.parse_args(); target_new=a.per_city-700
    root=a.root; image_dir=root/'images'
    base=pd.read_csv(root/'images_manifest_expanded_700.csv',dtype={'uuid':str,'orig_id':str,'sequence_id':str,'oecd_region_code':str},low_memory=False)
    bi=pd.read_csv(root/'image_integrity_expanded_700.csv',dtype={'uuid':str,'sha256':str},low_memory=False)
    init=pd.read_csv(a.initial_manifest,dtype={'uuid':str,'orig_id':str,'sequence_id':str,'oecd_region_code':str},low_memory=False)
    ii=pd.read_csv(a.initial_integrity,dtype={'uuid':str,'sha256':str},low_memory=False)
    if len(base)!=135*700 or base.city.nunique()!=135 or len(bi)!=len(base) or bi.sha256.nunique()!=len(bi):
        raise ValueError('The 135-city, 700-image base is incomplete or not content-unique')
    if len(init)!=135*target_new or init.groupby('city').size().ne(target_new).any():
        raise ValueError('Initial supplement manifest is not exactly 300 images/city')
    if init.uuid.duplicated().any() or set(ii.uuid)!=set(init.uuid) or ii.uuid.duplicated().any():
        raise ValueError('Initial supplement integrity audit does not exactly cover the manifest')
    initial_audit=ii.set_index('uuid')
    used_hashes=set(bi.sha256.astype(str))
    chosen_rows=[]; chosen_integrity=[]; deficits={}; chosen_ids=set()
    # Consume valid initial samples first, in their recorded randomized selection order.
    for city, group in init.groupby('city',sort=False):
        group=group.sort_values(['selection_rank_within_city','uuid'],kind='stable')
        for row in group.itertuples(index=False):
            ir=initial_audit.loc[str(row.uuid)]
            reason='' if pd.isna(ir.reason) else str(ir.reason)
            digest='' if pd.isna(ir.sha256) else str(ir.sha256)
            if reason or not digest or digest in used_hashes: continue
            chosen_rows.append((row._asdict(),digest,int(ir.width),int(ir.height)))
            chosen_ids.add(str(row.uuid)); used_hashes.add(digest)
        have=sum(1 for r,_,_,_ in chosen_rows if str(r['city'])==str(city))
        deficits[str(city)]=max(0,target_new-have)
    # Add accepted replacement candidates in sampling-tier order until every city reaches 300 new images.
    replacement_audits=[]
    for manifest_path in a.replacement_manifests:
        integrity_path=manifest_path.with_name(manifest_path.name.replace('replacements_','replacement_integrity_').replace('.csv','.csv'))
        if not integrity_path.is_file(): raise FileNotFoundError(integrity_path)
        cand=pd.read_csv(manifest_path,dtype={'uuid':str,'orig_id':str,'sequence_id':str,'oecd_region_code':str},low_memory=False)
        aud=pd.read_csv(integrity_path,dtype={'uuid':str,'sha256':str},low_memory=False)
        if cand.uuid.duplicated().any() or set(cand.uuid)!=set(aud.uuid) or aud.uuid.duplicated().any():
            raise ValueError(f'Candidate integrity mismatch: {manifest_path}')
        ab=aud.set_index('uuid')
        replacement_audits.append((manifest_path.name,len(cand),int((aud.reason.fillna('')=='').sum())))
        for row in cand.itertuples(index=False):
            city=str(row.city)
            if deficits.get(city,0)<=0: continue
            ir=ab.loc[str(row.uuid)]
            reason='' if pd.isna(ir.reason) else str(ir.reason)
            digest='' if pd.isna(ir.sha256) else str(ir.sha256)
            if reason or not digest or digest in used_hashes: continue
            chosen_rows.append((row._asdict(),digest,int(ir.width),int(ir.height)))
            chosen_ids.add(str(row.uuid)); used_hashes.add(digest); deficits[city]-=1
    remaining={c:n for c,n in deficits.items() if n>0}
    if remaining: raise RuntimeError(f'Insufficient valid, unique supplements by city: {remaining}')
    supplement=pd.DataFrame([r for r,_,_,_ in chosen_rows])
    if supplement.groupby('city').size().ne(target_new).any() or supplement.city.nunique()!=135:
        counts=supplement.groupby('city').size().to_dict()
        raise RuntimeError(f'Final supplement counts are not exactly {target_new}/city: {counts}')
    # Promote chosen images from staging and verify their hashes after the move.
    new_integrity=[]
    for row,digest,width,height in chosen_rows:
        uid=str(row['uuid']); src=a.supplement_dir/f'{uid}.jpg'; dst=image_dir/f'{uid}.jpg'
        if not dst.exists():
            if not src.is_file(): raise FileNotFoundError(src)
            dst.parent.mkdir(parents=True,exist_ok=True); src.replace(dst)
        got_w,got_h=verify(dst); got_hash=sha256(dst)
        if (got_w,got_h)!=(width,height) or got_hash!=digest:
            raise ValueError(f'Promoted image failed integrity check: {uid}')
        new_integrity.append({'uuid':uid,'city':row['city'],'mapillary_id':str(row['orig_id']),'bytes':dst.stat().st_size,'width':width,'height':height,'sha256':digest,'decode':'ok'})
    supplement=supplement.sort_values(['city','selection_rank_within_city','uuid'],kind='stable')
    final=pd.concat([base,supplement],ignore_index=True,sort=False)
    new_i=pd.DataFrame(new_integrity,columns=bi.columns)
    final_i=pd.concat([bi,new_i],ignore_index=True,sort=False)
    if len(final)!=135*a.per_city or final.city.nunique()!=135 or final.groupby('city').size().ne(a.per_city).any(): raise RuntimeError('Final manifest is not exactly 1,000 images in each of 135 cities')
    if final.uuid.duplicated().any() or final.orig_id.astype(str).duplicated().any() or final.sequence_id.astype(str).duplicated().any(): raise RuntimeError('Duplicate UUID, Mapillary image ID, or sequence in final manifest')
    if len(final_i)!=len(final) or final_i.sha256.nunique()!=len(final_i): raise RuntimeError('Final integrity rows are incomplete or content hashes are duplicated')
    a.output_manifest.parent.mkdir(parents=True,exist_ok=True); final.to_csv(a.output_manifest,index=False,encoding='utf-8-sig')
    a.output_integrity.parent.mkdir(parents=True,exist_ok=True); final_i.to_csv(a.output_integrity,index=False,encoding='utf-8-sig')
    summary={'final_images':len(final),'final_cities':int(final.city.nunique()),'images_per_city':a.per_city,'supplement_images':len(supplement),'unique_oecd_regions':int(final.oecd_region_code.nunique()),'unique_mapillary_ids':int(final.orig_id.astype(str).nunique()),'unique_sequences':int(final.sequence_id.astype(str).nunique()),'unique_sha256':int(final_i.sha256.nunique()),'image_dir_jpeg_count_after_promotion':len(list(image_dir.glob('*.jpg'))),'supplement_source_images_promoted':len(chosen_rows),'supplement_valid_unique_pool':int((ii.reason.fillna('')=='').sum()+sum(x[2] for x in replacement_audits)),'replacement_batches':replacement_audits,'manifest':str(a.output_manifest.resolve()),'integrity':str(a.output_integrity.resolve())}
    (root/'audit_expanded_1000.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
