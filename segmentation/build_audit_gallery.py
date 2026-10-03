#!/usr/bin/env python3
"""Build a lightweight local HTML gallery for human segmentation review."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/segmentation_audit"
TARGET_CLASSES = [
    "building", "wall", "road", "sidewalk", "tree", "grass", "plant", "sky",
    "car", "bus", "truck", "person", "fence", "signboard", "poster", "pole",
    "streetlight", "awning", "bench", "ashcan",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--max-preview", type=int, default=760)
    parser.add_argument("--masks-only", action="store_true", help="build a cloned gallery showing only colorized segmentation masks")
    parser.add_argument("--visual-subdir", default=None, help="output directory name under audit-dir")
    return parser.parse_args()


def make_preview(source: Path, destination: Path, max_size: int) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {source}")
    height, width = image.shape[:2]
    scale = min(max_size / max(height, width), 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 86])


def make_mask_preview(source: Path, destination: Path, palette_rgb: list[list[int]], max_size: int) -> None:
    mask = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read {source}")
    palette = np.asarray(palette_rgb, dtype=np.uint8)[:, ::-1]
    image = palette[np.clip(mask, 0, len(palette) - 1)]
    height, width = image.shape[:2]
    scale = min(max_size / max(height, width), 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_NEAREST)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 92])


def top_classes(row: pd.Series, limit: int = 8) -> str:
    values = []
    for key, value in row.items():
        if key.startswith("ratio__") and float(value) > 0:
            values.append((key.removeprefix("ratio__"), float(value)))
    return ", ".join(f"{html.escape(name)} {ratio:.1%}" for name, ratio in sorted(values, key=lambda x: x[1], reverse=True)[:limit])


def main() -> None:
    args = parse_args()
    audit = args.audit_dir.resolve()
    visual = audit / (args.visual_subdir or ("visual_review_masks" if args.masks_only else "visual_review"))
    manifest = pd.read_csv(audit / "audit_manifest.csv").sort_values("audit_order")
    comparison = pd.read_csv(audit / "baseline_vs_sliding.csv").set_index("image_id")
    baseline = pd.read_csv(audit / "baseline/semantic_features_raw.csv").set_index("image_id")
    sliding = pd.read_csv(audit / "sliding_512_o25/semantic_features_raw.csv").set_index("image_id")
    run_config = json.loads((audit / "run_config.json").read_text(encoding="utf-8"))
    palette = dict(zip(run_config["classes"], run_config["palette_rgb"]))
    target_classes = run_config.get("target_classes", TARGET_CLASSES)
    checkpoint_name = Path(run_config["checkpoint"]).name

    cards = []
    for _, row in manifest.iterrows():
        image_id = row["image_id"]
        if args.masks_only:
            sources = {"baseline": audit / "baseline/masks" / f"{image_id}.png", "sliding": audit / "sliding_512_o25/masks" / f"{image_id}.png"}
            for label, source in sources.items():
                make_mask_preview(source, visual / "previews" / label / f"{image_id}.jpg", run_config["palette_rgb"], args.max_preview)
            image_markup = (
                f'<figure><a href="../baseline/masks/{image_id}.png"><img loading="lazy" src="previews/baseline/{image_id}.jpg"></a><figcaption>缩放基线分割 mask</figcaption></figure>'
                f'<figure><a href="../sliding_512_o25/masks/{image_id}.png"><img loading="lazy" src="previews/sliding/{image_id}.jpg"></a><figcaption>512滑窗分割 mask（25%重叠）</figcaption></figure>'
            )
        else:
            sources = {"original": ROOT / row["image_path"], "baseline": audit / "baseline/overlays" / f"{image_id}.jpg", "sliding": audit / "sliding_512_o25/overlays" / f"{image_id}.jpg"}
            for label, source in sources.items():
                make_preview(source, visual / "previews" / label / f"{image_id}.jpg", args.max_preview)
            image_markup = (
                f'<figure><a href="../../../{html.escape(row["image_path"])}"><img loading="lazy" src="previews/original/{image_id}.jpg"></a><figcaption>原图</figcaption></figure>'
                f'<figure><a href="../baseline/overlays/{image_id}.jpg"><img loading="lazy" src="previews/baseline/{image_id}.jpg"></a><figcaption>缩放基线</figcaption></figure>'
                f'<figure><a href="../sliding_512_o25/overlays/{image_id}.jpg"><img loading="lazy" src="previews/sliding/{image_id}.jpg"></a><figcaption>512滑窗（25%重叠）</figcaption></figure>'
            )
        comp = comparison.loc[image_id]
        card = f"""
<article class="card" data-city="{html.escape(row['city'])}" data-stratum="{row['vp_stratum']}">
  <h2>#{int(row['audit_order']):02d} · {html.escape(row['city'])} · {row['vp_stratum'].upper()}</h2>
  <p class="meta"><code>{image_id}</code> · VP v8 {row['vp_score_v8']:.2f} · {int(row['width'])}×{int(row['height'])} · {html.escape(str(row['source']))} / {html.escape(str(row['platform']))} · pano={row['pano_status']}</p>
  <p class="metric">基线/滑窗像素一致率 <strong>{comp['pixel_agreement']:.1%}</strong>；方法间 mIoU <strong>{comp['mean_iou_between_methods']:.3f}</strong>；类别数 {int(comp['baseline_classes'])} → {int(comp['sliding_classes'])}</p>
  <div class="images">{image_markup}</div>
  <details><summary>面积最大的类别</summary><p><b>基线：</b>{top_classes(baseline.loc[image_id])}</p><p><b>滑窗：</b>{top_classes(sliding.loc[image_id])}</p></details>
  <div class="review" data-id="{image_id}">
    <label>基线质量 <select class="baseline"><option></option><option>0</option><option>1</option><option>2</option><option>NA</option></select></label>
    <label>滑窗质量 <select class="sliding"><option></option><option>0</option><option>1</option><option>2</option><option>NA</option></select></label>
    <label>偏好 <select class="preferred"><option></option><option>baseline</option><option>sliding</option><option>tie</option><option>neither</option></select></label>
    <label class="notes-label">备注 <input class="notes" placeholder="小目标遗漏、类别混淆、接缝等"></label>
  </div>
</article>"""
        cards.append(card)

    legend = "".join(
        f'<span><i style="background:rgb({palette[name][0]},{palette[name][1]},{palette[name][2]})"></i>{html.escape(name)}</span>'
        for name in target_classes
        if name in palette
    )
    cities = "".join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in sorted(manifest.city.unique()))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GUVP 分割人工审计{'（仅 mask）' if args.masks_only else ''}</title>
<style>
 :root{{--bg:#f4f5f7;--card:#fff;--ink:#20242a;--muted:#657080;--line:#d9dee6;--blue:#1769aa}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}header{{position:sticky;top:0;z-index:5;background:#fffffff2;border-bottom:1px solid var(--line);padding:14px 20px;backdrop-filter:blur(8px)}}h1{{font-size:21px;margin:0 0 8px}}.toolbar{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}select,button,input{{font:inherit;padding:6px 8px;border:1px solid #b8c0cc;border-radius:6px;background:white}}button{{background:var(--blue);color:white;border:0;cursor:pointer}}main{{max-width:1700px;margin:auto;padding:18px}}.help,.legend{{background:white;border:1px solid var(--line);border-radius:9px;padding:12px;margin-bottom:15px}}.legend{{display:flex;gap:11px;flex-wrap:wrap}}.legend span{{white-space:nowrap}}.legend i{{display:inline-block;width:13px;height:13px;margin-right:4px;vertical-align:-2px;border:1px solid #999}}.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 0 18px;box-shadow:0 2px 7px #0000000b}}h2{{font-size:18px;margin:0 0 3px}}.meta,.metric{{margin:3px 0;color:var(--muted)}}.images{{display:grid;grid-template-columns:repeat({'2' if args.masks_only else '3'},minmax(0,1fr));gap:9px;margin-top:10px}}figure{{margin:0;background:#eef0f3;border-radius:7px;overflow:hidden;text-align:center}}figure img{{width:100%;height:420px;object-fit:contain;display:block;background:#202124}}figcaption{{padding:5px;font-weight:650}}details{{margin-top:8px}}.review{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:10px;padding:9px;background:#f1f6fb;border-radius:7px}}.notes-label{{flex:1;min-width:300px}}.notes{{width:calc(100% - 45px)}}.hidden{{display:none}}@media(max-width:900px){{.images{{grid-template-columns:1fr}}figure img{{height:auto;max-height:650px}}}}
</style></head><body>
<header><h1>GUVP DeepLabv3+ 目标域人工审计（30图）{'— 仅显示分割 mask' if args.masks_only else ''}</h1><div class="toolbar"><label>城市 <select id="city"><option value="all">全部</option>{cities}</select></label><label>VP层 <select id="stratum"><option value="all">全部</option><option value="high">高</option><option value="mid">中</option><option value="low">低</option></select></label><button id="export">导出图像级评分 CSV</button><a href="../audit_ratings.csv">下载类别级评分模板</a></div><small>checkpoint: {html.escape(checkpoint_name)}</small></header>
<main><section class="help">评分：0=不可用/严重错误，1=部分可用，2=总体可靠，NA=无法判断。{'当前页面仅显示彩色分割 mask，不包含原始底图；点击 mask 可打开原始 mask。' if args.masks_only else '点击图片可打开原始分辨率。'}页面输入自动保存在当前浏览器；“导出”生成图像级评分文件。方法差异不代表滑窗一定更好，必须结合原图判断。</section><section class="legend">{legend}</section>{''.join(cards)}</main>
<script>
const key='guvp-seg-audit-{html.escape(visual.name)}'; const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
document.querySelectorAll('.review').forEach(r=>{{const id=r.dataset.id, v=saved[id]||{{}}; ['baseline','sliding','preferred','notes'].forEach(k=>{{r.querySelector('.'+k).value=v[k]||''}}); r.addEventListener('input',()=>{{saved[id]={{baseline:r.querySelector('.baseline').value,sliding:r.querySelector('.sliding').value,preferred:r.querySelector('.preferred').value,notes:r.querySelector('.notes').value}};localStorage.setItem(key,JSON.stringify(saved))}})}});
function filter(){{const c=document.querySelector('#city').value,s=document.querySelector('#stratum').value;document.querySelectorAll('.card').forEach(x=>x.classList.toggle('hidden',!(c==='all'||x.dataset.city===c)||!(s==='all'||x.dataset.stratum===s)))}} document.querySelector('#city').onchange=filter;document.querySelector('#stratum').onchange=filter;
document.querySelector('#export').onclick=()=>{{let rows=[['image_id','baseline_rating','sliding_rating','preferred_method','notes']];document.querySelectorAll('.review').forEach(r=>{{let v=saved[r.dataset.id]||{{}};rows.push([r.dataset.id,v.baseline||'',v.sliding||'',v.preferred||'',v.notes||''])}});let esc=x=>'"'+String(x).replaceAll('"','""')+'"';let blob=new Blob([rows.map(r=>r.map(esc).join(',')).join('\n')],{{type:'text/csv'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='segmentation_image_review.csv';a.click();URL.revokeObjectURL(a.href)}};
</script></body></html>"""
    visual.mkdir(parents=True, exist_ok=True)
    (visual / "index.html").write_text(document, encoding="utf-8")
    print(f"Wrote human-review gallery: {visual / 'index.html'}")


if __name__ == "__main__":
    main()
