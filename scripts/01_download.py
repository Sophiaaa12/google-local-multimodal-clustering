#!/usr/bin/env python3
"""Download official state files; streams bytes and supports HTTP Range resume."""
import argparse
import re
import sys
from pathlib import Path

import requests
from tqdm import tqdm

BASE = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal"
ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)\s*$", re.IGNORECASE)


def _parse_content_range(header):
    """Return (start, end, total) or None. total is None when the size is '*'."""
    if not header:
        return None
    match = _CONTENT_RANGE.match(header.strip())
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    total = None if match.group(3) == "*" else int(match.group(3))
    return start, end, total


def _content_length(headers):
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        length = int(raw)
    except ValueError:
        return None
    return length if length >= 0 else None


def _partial_size(path):
    return path.stat().st_size if path.exists() else 0


def _get(url, offset):
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(30, 120),
        allow_redirects=True,
    )


def _preserve_partial(partial, reason):
    size = _partial_size(partial)
    print(reason)
    if size:
        print(
            f"Partial download preserved at {partial} ({size:,} bytes). "
            "Re-run the same command to resume."
        )
    else:
        print("No data was written. Re-run the same command to retry.")
    raise SystemExit(1)


def download(state, kind, overwrite=False):
    prefix = "meta" if kind == "metadata" else "review"
    filename = f"{prefix}-{state}.json.gz"
    dest = ROOT / "data" / "raw" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        print(f"Already exists: {dest} (use --overwrite to re-download)")
        return
    partial = dest.with_name(dest.name + ".part")
    offset = _partial_size(partial) if partial.exists() and not overwrite else 0
    url = f"{BASE}/{filename}"
    response = None
    try:
        response = _get(url, offset)
        response.raise_for_status()
        if offset:
            parsed = _parse_content_range(response.headers.get("Content-Range"))
            honored = (
                response.status_code == 206
                and parsed is not None
                and parsed[0] == offset
            )
            if honored:
                total = parsed[2]
                if total is None:
                    remaining = _content_length(response.headers)
                    total = offset + remaining if remaining is not None else None
            else:
                print(
                    "Server did not honor Range "
                    "(expected HTTP 206 with a matching Content-Range); "
                    "restarting download"
                )
                response.close()
                response = None
                offset = 0
                response = _get(url, 0)
                response.raise_for_status()
                total = _content_length(response.headers)
        else:
            total = _content_length(response.headers)

        mode = "ab" if offset else "wb"
        written_start = offset
        with partial.open(mode) as handle, tqdm(
            desc=filename,
            total=total,
            initial=offset,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
        ) as bar:
            for chunk in response.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                handle.write(chunk)
                bar.update(len(chunk))
    except requests.HTTPError:
        raise
    except requests.Timeout as exc:
        _preserve_partial(partial, f"Download timed out: {exc}")
    except requests.RequestException as exc:
        _preserve_partial(partial, f"Download interrupted: {exc}")
    except KeyboardInterrupt:
        _preserve_partial(partial, "Download interrupted by user.")
    finally:
        if response is not None:
            response.close()

    size = _partial_size(partial)
    if total is not None and size != total:
        print(
            f"Incomplete download: received {size:,} of {total:,} bytes. "
            f"Partial file preserved at {partial}."
        )
        raise SystemExit(1)
    if size <= written_start and total is None and offset:
        print(
            f"No new bytes received. Partial file preserved at {partial} ({size:,} bytes)."
        )
        raise SystemExit(1)
    partial.replace(dest)
    print(f"Downloaded {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--state", default="New_York", help="Use underscores, e.g. New_York")
    p.add_argument("--kind", choices=["metadata", "reviews", "both"], default="metadata")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    try:
        for kind in (["metadata", "reviews"] if a.kind == "both" else [a.kind]):
            download(a.state, kind, a.overwrite)
    except requests.HTTPError as exc:
        print(f"Download failed: {exc}")
        sys.exit(1)
