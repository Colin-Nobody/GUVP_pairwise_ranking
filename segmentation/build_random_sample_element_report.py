#!/usr/bin/env python3
"""Create an element-percentage report for the random segmentation sample."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir, sample_dir = args.run_dir.resolve(), args.sample_dir.resolve()
    selected = pd.read_csv(sample_dir / "selected_30.csv")
    percentages = pd.read_csv(run_dir / "component_percentages_long.csv")
    sample_ids = set(selected["image_id"])
    report = percentages[percentages["image_id"].isin(sample_ids)].copy()
    report = report.merge(selected[["image_id", "sample_number", "city", "triptych"]], on="image_id", how="left")
    report = report.sort_values(["sample_number", "pixel_percentage", "class_name"], ascending=[True, False, True])
    csv_path = sample_dir / "element_percentages_30_long.csv"
    report.to_csv(csv_path, index=False)

    config = json.loads((run_dir / "run_config.json").read_text())
    colors = {name: rgb for name, rgb in zip(config["classes"], config["palette_rgb"])}
    cards: list[str] = []
    for _, image in selected.sort_values("sample_number").iterrows():
        rows = report[report["image_id"] == image["image_id"]]
        top = rows[rows["pixel_percentage"] > 0].head(12)
        chips = "".join(
            f'<span class="chip"><i style="background:rgb({colors[row.class_name][0]},{colors[row.class_name][1]},{colors[row.class_name][2]})"></i>{html.escape(row.class_name)} {row.pixel_percentage:.2f}%</span>'
            for row in top.itertuples()
        ) or '<span class="chip">No predicted class</span>'
        table = "".join(
            f'<tr><td><i style="background:rgb({colors[row.class_name][0]},{colors[row.class_name][1]},{colors[row.class_name][2]})"></i>{html.escape(row.class_name)}</td><td>{row.pixel_percentage:.4f}%</td></tr>'
            for row in rows.itertuples()
        )
        cards.append(f'''<article><h2>#{int(image.sample_number):02d} · {html.escape(str(image.city))} · <code>{html.escape(image.image_id)}</code></h2>
<img src="{html.escape(image.triptych)}" alt="Original, colour mask, and overlay for {html.escape(image.image_id)}">
<p class="top"><b>出现面积最大的元素：</b>{chips}</p>
<details><summary>查看全部 124 个元素的像素占比（含 0%）</summary><table><thead><tr><th>Element</th><th>Percent</th></tr></thead><tbody>{table}</tbody></table></details></article>''')
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>30 张随机样本：语义元素占比</title>
<style>body{{margin:0;background:#f3f5f7;color:#20242a;font:15px/1.5 system-ui,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#ffffffed;border-bottom:1px solid #d9dee6;padding:13px 22px;backdrop-filter:blur(8px)}}h1{{margin:0;font-size:21px}}header p{{margin:4px 0 0;color:#596575}}main{{max-width:2250px;margin:auto;padding:20px}}article{{background:#fff;margin:0 0 20px;padding:14px;border:1px solid #d9dee6;border-radius:10px;box-shadow:0 2px 7px #0000000b}}h2{{font-size:17px;margin:0 0 8px}}img{{display:block;width:100%;max-width:2160px;background:#202124}}.top{{display:flex;gap:7px;flex-wrap:wrap;align-items:center}}.chip{{display:inline-flex;align-items:center;background:#eef2f6;border-radius:14px;padding:2px 8px;white-space:nowrap}}i{{display:inline-block;width:13px;height:13px;border:1px solid #7a8491;margin-right:5px;vertical-align:-2px}}details{{margin-top:10px}}table{{border-collapse:collapse;margin-top:8px;min-width:440px}}th,td{{border:1px solid #d9dee6;padding:4px 9px;text-align:left}}th{{background:#f0f3f6}}td:nth-child(2){{text-align:right;font-variant-numeric:tabular-nums}}</style></head><body>
<header><h1>随机 30 张：原图 / 彩色语义分割 / 叠加图与元素占比</h1><p>中栏颜色与元素名称对应。百分比按该图像全部像素计算；展开可查看全部 124 类。</p></header><main>{''.join(cards)}</main></body></html>'''
    html_path = sample_dir / "index_with_element_percentages.html"
    html_path.write_text(document, encoding="utf-8")
    print(f"Wrote {len(report)} rows: {csv_path}")
    print(f"Wrote HTML report: {html_path}")


if __name__ == "__main__":
    main()
