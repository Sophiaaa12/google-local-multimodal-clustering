#!/usr/bin/env python3
"""Sample NYC businesses and stream a deterministic review subset."""
import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
BUSINESS_FIELDS = (
    "gmap_id",
    "name",
    "category",
    "description",
    "latitude",
    "longitude",
    "avg_rating",
    "num_of_reviews",
)
REVIEW_FIELDS = ("gmap_id", "text", "pics", "rating", "time")


def in_nyc(lat, lon):
    return isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and 40.48 <= lat <= 40.93 and -74.26 <= lon <= -73.68


def _compressed_pos(handle):
    return handle.buffer.fileobj.tell()


def _scan(path):
    """Stream a .json.gz file and report compressed bytes read."""
    handle = gzip.open(path, "rt", encoding="utf-8")
    bar = tqdm(
        total=path.stat().st_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=path.name,
        miniters=1,
    )
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


def choose_businesses(ids, sample_size, seed):
    """Deterministic sample of gmap_ids. Returns ids in sorted order."""
    ordered = sorted(ids)
    if sample_size < 0:
        raise SystemExit("--sample-size must be >= 0")
    k = min(sample_size, len(ordered))
    if k == 0:
        return []
    picked = random.Random(seed).sample(ordered, k)
    return sorted(picked)


def _review_rng(seed, gmap_id):
    digest = hashlib.sha256(f"{seed}:{gmap_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def image_urls(pics):
    """Flatten the Google Local pics field into URL strings."""
    urls = []
    if not pics:
        return urls
    items = pics if isinstance(pics, list) else [pics]
    for item in items:
        if isinstance(item, str):
            urls.append(item)
        elif isinstance(item, dict):
            raw = item.get("url")
            if isinstance(raw, str):
                urls.append(raw)
            elif isinstance(raw, list):
                urls.extend(u for u in raw if isinstance(u, str))
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


class ReviewReservoir:
    """Keep at most k reviews, chosen uniformly as the file is streamed."""

    def __init__(self, k, rng):
        self.k = k
        self.rng = rng
        self.items = []
        self.seen = 0

    def add(self, seq, review):
        self.seen += 1
        if self.k <= 0:
            return
        record = (seq, review)
        if len(self.items) < self.k:
            self.items.append(record)
            return
        slot = self.rng.randrange(self.seen)
        if slot < self.k:
            self.items[slot] = record

    def chosen(self):
        return [review for _, review in sorted(self.items, key=lambda pair: pair[0])]


def main(state, sample_size, max_reviews, seed):
    raw = ROOT / "data" / "raw"
    meta_path, review_path = raw / f"meta-{state}.json.gz", raw / f"review-{state}.json.gz"
    for path in (meta_path, review_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}; download it first")
    eligible = {}
    for line in _scan(meta_path):
        item = json.loads(line)
        if in_nyc(item.get("latitude"), item.get("longitude")) and item.get("gmap_id"):
            eligible[item["gmap_id"]] = item
    selected_ids = choose_businesses(eligible, sample_size, seed)
    selected = {gmap_id: eligible[gmap_id] for gmap_id in selected_ids}
    reservoirs = {
        gmap_id: ReviewReservoir(max_reviews, _review_rng(seed, gmap_id))
        for gmap_id in selected_ids
    }

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    business_file = out / f"businesses_{state}.jsonl"
    review_file = out / f"reviews_{state}.jsonl"
    counts = Counter()
    with business_file.open("w", encoding="utf-8") as handle:
        for gmap_id in selected_ids:
            item = selected[gmap_id]
            safe = {key: item.get(key) for key in BUSINESS_FIELDS}
            handle.write(json.dumps(safe, ensure_ascii=False) + "\n")

    seq = 0
    for line in _scan(review_path):
        item = json.loads(line)
        key = item.get("gmap_id")
        if key not in selected:
            continue
        counts["matched_reviews"] += 1
        has_text = bool(item.get("text"))
        urls = image_urls(item.get("pics"))
        if has_text:
            counts["matched_with_text"] += 1
        if urls:
            counts["matched_with_pics"] += 1
            counts["matched_image_urls"] += len(urls)
        if has_text and urls:
            counts["matched_with_text_and_pics"] += 1
        if not has_text and not urls:
            counts["matched_empty"] += 1
            continue
        seq += 1
        reservoirs[key].add(seq, {field: item.get(field) for field in REVIEW_FIELDS})

    kept_image_urls = set()
    with review_file.open("w", encoding="utf-8") as dest:
        for gmap_id in selected_ids:
            for review in reservoirs[gmap_id].chosen():
                counts["kept_reviews"] += 1
                if review.get("text"):
                    counts["kept_with_text"] += 1
                urls = image_urls(review.get("pics"))
                if urls:
                    counts["kept_with_pics"] += 1
                    counts["kept_image_urls"] += len(urls)
                    kept_image_urls.update(urls)
                if review.get("text") and urls:
                    counts["kept_with_text_and_pics"] += 1
                dest.write(json.dumps(review, ensure_ascii=False) + "\n")

    businesses_with_kept = sum(1 for reservoir in reservoirs.values() if reservoir.chosen())
    summary = {
        "state": state,
        "eligible_nyc_bbox_businesses": len(eligible),
        "selected_businesses": len(selected),
        "matched_reviews": counts["matched_reviews"],
        "matched_with_text": counts["matched_with_text"],
        "matched_with_pics": counts["matched_with_pics"],
        "matched_with_text_and_pics": counts["matched_with_text_and_pics"],
        "matched_empty": counts["matched_empty"],
        "matched_image_urls": counts["matched_image_urls"],
        "kept_reviews": counts["kept_reviews"],
        "kept_with_text": counts["kept_with_text"],
        "kept_with_pics": counts["kept_with_pics"],
        "kept_with_text_and_pics": counts["kept_with_text_and_pics"],
        "kept_image_urls": counts["kept_image_urls"],
        "unique_kept_image_urls": len(kept_image_urls),
        "businesses_with_kept_reviews": businesses_with_kept,
        "businesses_without_kept_reviews": len(selected) - businesses_with_kept,
        "sample_size": sample_size,
        "max_reviews_per_business": max_reviews,
        "seed": seed,
        "sampling": "uniform random sample of eligible NYC gmap_ids; reservoir sample of reviews with text or images, at most max_reviews_per_business, seeded per business",
    }
    summary_file = ROOT / "results" / "tables" / f"coverage_{state}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="New_York")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument(
        "--max-businesses",
        type=int,
        default=None,
        help="Override --sample-size. Kept so earlier commands still run.",
    )
    parser.add_argument("--max-reviews-per-business", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    size = args.sample_size if args.max_businesses is None else args.max_businesses
    main(args.state, size, args.max_reviews_per_business, args.seed)
