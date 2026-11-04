#!/usr/bin/env python3
"""Sample review image URLs and check that they decode as images."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import random
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]


def _load_image_urls():
    spec = importlib.util.spec_from_file_location("preprocess", ROOT / "scripts" / "03_preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.image_urls


image_urls = _load_image_urls()


def collect_urls(review_path):
    urls = []
    with review_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            urls.extend(image_urls(json.loads(line).get("pics")))
    return urls


def sample_urls(urls, sample_size, seed):
    unique = sorted(set(urls))
    if sample_size < 0:
        raise SystemExit("--sample-size must be >= 0")
    k = min(sample_size, len(unique))
    if k == 0:
        return []
    return random.Random(seed).sample(unique, k)


def _session():
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "google-local-multimodal-clustering/image-check"
    return session


def _filename(url, content_type):
    suffix = ".img"
    lowered = (content_type or "").split(";")[0].strip().lower()
    known = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }
    suffix = known.get(lowered, suffix)
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    return digest + suffix


def check_url(session, url, dest_dir):
    row = {
        "url": url,
        "http_status": "",
        "success": False,
        "width": "",
        "height": "",
        "file_size_bytes": "",
        "error": "",
        "saved_path": "",
    }
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        row["error"] = "invalid_url"
        return row
    try:
        response = session.get(url, timeout=(10, 20), stream=True)
    except requests.RequestException as exc:
        row["error"] = f"request_failed: {exc.__class__.__name__}"
        return row
    row["http_status"] = response.status_code
    try:
        data = response.content
    finally:
        response.close()
    row["file_size_bytes"] = len(data)
    if response.status_code != 200:
        row["error"] = f"http_{response.status_code}"
        return row
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            fmt = image.format
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        row["error"] = f"not_an_image: {exc.__class__.__name__}"
        return row
    if width <= 0 or height <= 0:
        row["error"] = "empty_image"
        return row
    content_type = response.headers.get("Content-Type", "")
    if not content_type and fmt:
        content_type = f"image/{fmt.lower()}"
    path = dest_dir / _filename(url, content_type)
    path.write_bytes(data)
    row["success"] = True
    row["width"] = width
    row["height"] = height
    row["saved_path"] = str(path)
    return row


def main(state, sample_size, seed, review_path=None):
    reviews = Path(review_path) if review_path else ROOT / "data" / "processed" / f"reviews_{state}.jsonl"
    if not reviews.exists():
        raise SystemExit(f"Missing {reviews}; run scripts/03_preprocess.py first")
    urls = collect_urls(reviews)
    chosen = sample_urls(urls, sample_size, seed)
    dest_dir = ROOT / "data" / "processed" / "sample_images"
    dest_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "results" / "tables" / f"image_validation_{state}.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "url",
        "http_status",
        "success",
        "width",
        "height",
        "file_size_bytes",
        "error",
        "saved_path",
    ]
    session = _session()
    rows = []
    for url in tqdm(chosen, desc="validate images", unit="img"):
        rows.append(check_url(session, url, dest_dir))
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(1 for row in rows if row["success"])
    print(
        json.dumps(
            {
                "state": state,
                "review_file": str(reviews),
                "image_urls_in_reviews": len(urls),
                "unique_image_urls": len(set(urls)),
                "sampled": len(chosen),
                "decoded": ok,
                "failed": len(chosen) - ok,
                "seed": seed,
                "report": str(report_path),
                "image_dir": str(dest_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="New_York")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reviews", default=None, help="Override the processed reviews JSONL path")
    args = parser.parse_args()
    main(args.state, args.sample_size, args.seed, args.reviews)
