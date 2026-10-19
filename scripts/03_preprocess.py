#!/usr/bin/env python3
"""Deterministically select NYC bbox businesses and stream matching reviews."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

def in_nyc(lat, lon):
    return isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and 40.48 <= lat <= 40.93 and -74.26 <= lon <= -73.68

def _compressed_pos(handle):
    return handle.buffer.fileobj.tell()

def _scan(path):
    """Stream a .json.gz file and report compressed bytes read."""
    handle = gzip.open(path, "rt", encoding="utf-8")
    bar = tqdm(total=path.stat().st_size, unit="B", unit_scale=True, unit_divisor=1024, desc=path.name, miniters=1)
    seen = 0

    def lines():
        nonlocal seen
        try:
            for line in handle:
                pos = _compressed_pos(handle)
                if pos > seen:
                    bar.update(pos - seen)
                    seen = pos
                yield line
        finally:
            pos = _compressed_pos(handle)
            if pos > seen:
                bar.update(pos - seen)
            bar.close()
            handle.close()

    return lines()

def main(state, max_businesses, max_reviews):
    raw = ROOT / "data" / "raw"
    meta_path, review_path = raw / f"meta-{state}.json.gz", raw / f"review-{state}.json.gz"
    for p in (meta_path, review_path):
        if not p.exists():
            raise SystemExit(f"Missing {p}; download it first")
    eligible = {}
    for line in _scan(meta_path):
        item = json.loads(line)
        if in_nyc(item.get("latitude"), item.get("longitude")) and item.get("gmap_id"):
            eligible[item["gmap_id"]] = item
    selected = {k: eligible[k] for k in sorted(eligible)[:max_businesses]}
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    business_file = out / f"businesses_{state}.jsonl"
    review_file = out / f"reviews_{state}.jsonl"
    counts = Counter()
    with business_file.open("w", encoding="utf-8") as f:
        for item in selected.values():
            safe = {k: item.get(k) for k in ("gmap_id", "name", "category", "description", "latitude", "longitude", "avg_rating", "num_of_reviews")}
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
    with review_file.open("w", encoding="utf-8") as dest:
        for line in _scan(review_path):
            item = json.loads(line)
            key = item.get("gmap_id")
            if key not in selected:
                continue
            counts["matched_reviews"] += 1
            if item.get("text"):
                counts["matched_with_text"] += 1
            if item.get("pics"):
                counts["matched_with_pics"] += 1
            if counts[("kept", key)] >= max_reviews:
                continue
            if not item.get("text") and not item.get("pics"):
                continue
            counts[("kept", key)] += 1
            counts["kept_reviews"] += 1
            safe = {k: item.get(k) for k in ("gmap_id", "text", "pics", "rating", "time")}
            dest.write(json.dumps(safe, ensure_ascii=False) + "\n")
    summary = {"state": state, "eligible_nyc_bbox_businesses": len(eligible), "selected_businesses": len(selected), "matched_reviews": counts["matched_reviews"], "matched_with_text": counts["matched_with_text"], "matched_with_pics": counts["matched_with_pics"], "kept_reviews": counts["kept_reviews"], "sampling": "first sorted gmap_ids; capped reviews per business; not representative"}
    summary_file = ROOT / "results" / "tables" / f"coverage_{state}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--state", default="New_York")
    p.add_argument("--max-businesses", type=int, default=5000)
    p.add_argument("--max-reviews-per-business", type=int, default=20)
    a = p.parse_args()
    main(a.state, a.max_businesses, a.max_reviews_per_business)
