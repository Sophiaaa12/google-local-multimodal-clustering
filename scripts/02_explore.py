#!/usr/bin/env python3
"""Stream metadata and summarize coordinate and NYC bbox coverage."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def in_nyc(lat, lon):
    return 40.48 <= lat <= 40.93 and -74.26 <= lon <= -73.68

def main(state):
    path = ROOT / "data" / "raw" / f"meta-{state}.json.gz"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run 01_download.py first")
    counts = Counter()
    categories = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue
            counts["businesses"] += 1
            lat, lon = item.get("latitude"), item.get("longitude")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                counts["with_coordinates"] += 1
                if in_nyc(lat, lon):
                    counts["in_nyc_bbox"] += 1
                    if item.get("description"):
                        counts["nyc_with_description"] += 1
                    for cat in item.get("category") or []:
                        categories[cat] += 1
    result = {"state": state, "counts": dict(counts), "nyc_top_categories": categories.most_common(20), "note": "NYC approximate bbox, not administrative boundary"}
    out = ROOT / "results" / "tables" / f"metadata_summary_{state}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--state", default="New_York")
    a = p.parse_args()
    main(a.state)
