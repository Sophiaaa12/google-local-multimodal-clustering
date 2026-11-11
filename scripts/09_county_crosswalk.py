#!/usr/bin/env python3
"""Assign Census county FIPS codes to sampled businesses.

Reads data/processed/businesses_{state}.jsonl and does not modify it.
Downloads the 2026 TIGER/Line national county file if the shapefile is not
already on disk, then point-in-polygon matches each coordinate.

Business latitude/longitude have no CRS in the JSONL. They are assumed to be
EPSG:4326 (WGS84), the same convention Google uses, and are transformed into
the shapefile CRS before the test. A point is matched only when exactly one
county polygon covers it, including the boundary. Points outside every polygon
stay unmatched. Points covered by more than one county stay ambiguous. Nothing
is snapped to the nearest county, and no row is dropped.
"""
import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import requests
import shapefile
from pyproj import CRS, Transformer
from shapely import STRtree, make_valid
from shapely.geometry import Point, shape
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2026/COUNTY/tl_2026_us_county.zip"
TIGER_ZIP = ROOT / "data" / "raw" / "tl_2026_us_county.zip"
TIGER_DIR = ROOT / "data" / "raw" / "tl_2026_us_county"
FIPS5 = re.compile(r"^\d{5}$")
FIPS2 = re.compile(r"^\d{2}$")
# Census county FIPS for the five New York City boroughs. Used only to label
# the summary. Matching itself does not filter on this set.
NYC_FIVE_BOROUGHS = {
    "36005": "Bronx",
    "36047": "Kings",
    "36061": "New York",
    "36081": "Queens",
    "36085": "Richmond",
}
# Display labels for the summary only. Not used to accept or reject a match.
STATE_FIPS_NAMES = {
    "09": "Connecticut",
    "34": "New Jersey",
    "36": "New York",
    "42": "Pennsylvania",
}
CANDIDATE_PAD_DEGREES = 0.02
CROSSWALK_FIELDS = (
    "gmap_id",
    "county_fips",
    "county_name",
    "state_fips",
    "match_status",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def download_tiger(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"Already exists: {dest}")
        return
    partial = dest.with_suffix(dest.suffix + ".part")
    response = requests.get(
        url,
        stream=True,
        timeout=(30, 180),
        headers={"User-Agent": "google-local-multimodal-clustering/county-crosswalk"},
    )
    response.raise_for_status()
    total = int(response.headers.get("Content-Length") or 0) or None
    try:
        with partial.open("wb") as handle, tqdm(
            desc=dest.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                bar.update(len(chunk))
    finally:
        response.close()
    if not zipfile.is_zipfile(partial):
        raise SystemExit(f"Download is not a zip file: {partial}")
    partial.replace(dest)
    print(f"Downloaded {dest} ({dest.stat().st_size:,} bytes)")


def extract_tiger(zip_path, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = list(dest_dir.rglob("tl_2026_us_county.shp"))
    if existing and existing[0].with_suffix(".prj").exists() and existing[0].with_suffix(".dbf").exists():
        return existing[0]
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)
    found = list(dest_dir.rglob("tl_2026_us_county.shp"))
    if len(found) != 1:
        raise SystemExit(f"Expected one tl_2026_us_county.shp under {dest_dir}, found {len(found)}")
    return found[0]


def load_businesses(path):
    rows = []
    seen = set()
    duplicate_ids = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gmap_id = item.get("gmap_id")
            if gmap_id in seen:
                duplicate_ids.append(gmap_id)
            elif gmap_id:
                seen.add(gmap_id)
            rows.append(
                {
                    "gmap_id": gmap_id,
                    "latitude": item.get("latitude"),
                    "longitude": item.get("longitude"),
                    "line": line_no,
                }
            )
    return rows, duplicate_ids


def valid_coordinate(lat, lon):
    return (
        isinstance(lat, (int, float))
        and not isinstance(lat, bool)
        and isinstance(lon, (int, float))
        and not isinstance(lon, bool)
        and math.isfinite(lat)
        and math.isfinite(lon)
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    )


def digit_code(value, width):
    """Return (code, coerced). coerced is true when padding or int formatting was required."""
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() and len(text) <= width:
            return text.zfill(width), len(text) != width
        return None, False
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        text = str(value)
        if len(text) <= width:
            return text.zfill(width), True
    return None, False


def load_crs(prj_path):
    crs = CRS.from_wkt(prj_path.read_text(encoding="utf-8"))
    epsg = crs.to_epsg()
    reference = CRS.from_epsg(4269)
    # TIGER writes longitude then latitude. EPSG:4269's official axis order is
    # latitude then longitude, so a strict equals() check is false even though
    # to_epsg() is 4269. Ignore axis order for the identity check, and use
    # always_xy later so values stay in the shapefile's longitude/latitude order.
    try:
        equals_4269 = bool(crs.equals(reference, ignore_axis_order=True))
    except Exception:
        equals_4269 = False
    return crs, epsg, equals_4269, reference


def load_candidate_counties(shp_path, bounds):
    cpg = shp_path.with_suffix(".cpg")
    encoding = "utf-8"
    if cpg.exists():
        raw = cpg.read_text(encoding="ascii", errors="replace").strip()
        if raw:
            encoding = raw
    reader = shapefile.Reader(str(shp_path), encoding=encoding)
    field_names = [field[0] for field in reader.fields[1:]]
    required = ("GEOID", "NAME", "STATEFP")
    missing = [name for name in required if name not in field_names]
    if missing:
        reader.close()
        raise SystemExit(f"Shapefile is missing fields {missing}. Found {field_names}")
    minx, miny, maxx, maxy = bounds
    pad = CANDIDATE_PAD_DEGREES
    candidates = []
    n_records = 0
    n_repaired = 0
    n_bad_geoid = 0
    n_geoid_state_mismatch = 0
    n_empty = 0
    try:
        for item in reader.iterShapeRecords():
            n_records += 1
            bbox = item.shape.bbox
            if not bbox:
                n_empty += 1
                continue
            outside = bbox[2] < minx - pad or bbox[0] > maxx + pad or bbox[3] < miny - pad or bbox[1] > maxy + pad
            if outside:
                continue
            geoid, _ = digit_code(item.record["GEOID"], 5)
            statefp, _ = digit_code(item.record["STATEFP"], 2)
            name = item.record["NAME"]
            if geoid is None or not FIPS5.fullmatch(geoid) or not isinstance(name, str) or not name.strip():
                n_bad_geoid += 1
                continue
            if statefp != geoid[:2]:
                n_geoid_state_mismatch += 1
            geom = shape(item.shape.__geo_interface__)
            if geom.is_empty:
                n_empty += 1
                continue
            if not geom.is_valid:
                geom = make_valid(geom)
                n_repaired += 1
                if geom.is_empty:
                    n_empty += 1
                    continue
            candidates.append(
                {
                    "county_fips": geoid,
                    "county_name": name.strip(),
                    "state_fips": geoid[:2],
                    "geom": geom,
                }
            )
    finally:
        reader.close()
    stats = {
        "n_county_records": n_records,
        "n_candidate_counties": len(candidates),
        "invalid_geometries_repaired": n_repaired,
        "records_skipped_bad_geoid": n_bad_geoid,
        "records_with_statefp_geoid_mismatch": n_geoid_state_mismatch,
        "empty_shapes_skipped_among_scanned": n_empty,
    }
    return candidates, stats


def assign(rows, transformer, candidates):
    geoms = [item["geom"] for item in candidates]
    for item in candidates:
        probe = item["geom"].representative_point()
        if not item["geom"].covers(probe):
            raise SystemExit(f"covers() failed on a representative point of county {item['county_fips']}")
    # query() without a predicate returns bounding-box hits only. covers() is
    # applied explicitly so the test is county.covers(point), including the boundary.
    tree = STRtree(geoms) if geoms else None
    assigned = []
    transformed = []
    for row in rows:
        lat, lon = row["latitude"], row["longitude"]
        base = {
            "gmap_id": row["gmap_id"],
            "county_fips": None,
            "county_name": None,
            "state_fips": None,
            "match_status": None,
            "latitude": lat,
            "longitude": lon,
            "candidates": [],
        }
        if not valid_coordinate(lat, lon):
            base["match_status"] = "invalid_coordinates"
            assigned.append(base)
            continue
        x, y = transformer.transform(lon, lat)
        transformed.append((x, y))
        point = Point(x, y)
        found = {}
        if tree is not None:
            for idx in tree.query(point):
                item = candidates[int(idx)]
                if item["geom"].covers(point):
                    found[item["county_fips"]] = item
        chosen = sorted(found.values(), key=lambda item: item["county_fips"])
        base["candidates"] = [
            {
                "county_fips": item["county_fips"],
                "county_name": item["county_name"],
                "state_fips": item["state_fips"],
            }
            for item in chosen
        ]
        if len(chosen) == 1:
            hit = chosen[0]
            base["county_fips"] = hit["county_fips"]
            base["county_name"] = hit["county_name"]
            base["state_fips"] = hit["state_fips"]
            base["match_status"] = "matched"
        elif len(chosen) == 0:
            base["match_status"] = "unmatched"
        else:
            base["match_status"] = "ambiguous"
        assigned.append(base)
    return assigned, transformed


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def check(checks, name, passed, detail):
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def build_checks(rows, duplicate_ids, assigned, transformed, crs_info, businesses_path, businesses_hash, written):
    checks = []
    input_ids = [row["gmap_id"] for row in rows]
    check(checks, "input_row_count", len(rows) == 10000, f"{len(rows)} rows")
    check(
        checks,
        "unique_gmap_id",
        len(duplicate_ids) == 0 and len(input_ids) == len(set(input_ids)) and all(isinstance(gid, str) and gid for gid in input_ids),
        f"duplicates={len(duplicate_ids)}",
    )
    check(checks, "tiger_crs_is_geographic", crs_info["tiger_is_geographic"], crs_info["tiger_crs_name"])
    check(
        checks,
        "tiger_crs_is_nad83",
        crs_info["tiger_epsg"] == 4269 and crs_info["tiger_equals_epsg_4269_ignoring_axis_order"],
        (
            f"epsg={crs_info['tiger_epsg']}, "
            f"strict_equals={crs_info['tiger_strict_equals_epsg_4269']}, "
            f"tiger_axis={crs_info['tiger_axis_order']}, "
            f"epsg_axis={crs_info['epsg_4269_axis_order']}"
        ),
    )
    if transformed:
        xs = [pair[0] for pair in transformed]
        ys = [pair[1] for pair in transformed]
        lons = [row["longitude"] for row in rows if valid_coordinate(row["latitude"], row["longitude"])]
        lats = [row["latitude"] for row in rows if valid_coordinate(row["latitude"], row["longitude"])]
        axis_ok = abs(median(xs) - median(lons)) < 1 and abs(median(ys) - median(lats)) < 1
        geographic_range = max(abs(x) for x in xs) <= 180 and max(abs(y) for y in ys) <= 90
        check(
            checks,
            "transformed_axis_matches_longitude_latitude",
            axis_ok and geographic_range,
            f"median_xy=({median(xs):.5f}, {median(ys):.5f}) median_lonlat=({median(lons):.5f}, {median(lats):.5f})",
        )
    else:
        check(checks, "transformed_axis_matches_longitude_latitude", False, "no valid coordinates were transformed")
    output_ids = [row["gmap_id"] for row in written]
    check(checks, "output_row_count", len(written) == len(rows), f"output={len(written)} input={len(rows)}")
    check(checks, "output_gmap_id_order", output_ids == input_ids, "same ids in the same order as the business file")
    statuses = {"matched": 0, "unmatched": 0, "ambiguous": 0, "invalid_coordinates": 0}
    fips_ok = True
    nulls_ok = True
    keys_ok = True
    candidate_fips = set()
    for row in assigned:
        for item in row["candidates"]:
            candidate_fips.add(item["county_fips"])
    for row in written:
        if tuple(row) != CROSSWALK_FIELDS:
            keys_ok = False
        status = row["match_status"]
        if status not in statuses:
            statuses.setdefault(status, 0)
        statuses[status] = statuses.get(status, 0) + 1
        if status == "matched":
            fips = row["county_fips"]
            state_fips = row["state_fips"]
            name = row["county_name"]
            if (
                not isinstance(fips, str)
                or FIPS5.fullmatch(fips) is None
                or not isinstance(state_fips, str)
                or FIPS2.fullmatch(state_fips) is None
                or fips[:2] != state_fips
                or not isinstance(name, str)
                or not name.strip()
                or fips not in candidate_fips
            ):
                fips_ok = False
        elif status in {"unmatched", "ambiguous", "invalid_coordinates"}:
            if row["county_fips"] is not None or row["county_name"] is not None or row["state_fips"] is not None:
                nulls_ok = False
        else:
            fips_ok = False
    check(checks, "match_status_partition", sum(statuses.get(name, 0) for name in ("matched", "unmatched", "ambiguous", "invalid_coordinates")) == len(written), str(statuses))
    check(checks, "matched_fips_codes", fips_ok, "5-digit county FIPS, 2-digit state FIPS, state prefix, TIGER name")
    check(checks, "unmatched_fields_are_null", nulls_ok, "unmatched, ambiguous, and invalid rows have null county fields")
    check(checks, "crosswalk_columns", keys_ok, ", ".join(CROSSWALK_FIELDS))
    check(
        checks,
        "businesses_file_unchanged",
        sha256(businesses_path) == businesses_hash,
        businesses_hash,
    )
    check(checks, "at_least_one_match", statuses.get("matched", 0) > 0, f"matched={statuses.get('matched', 0)}")
    return checks


def state_label(state_fips):
    name = STATE_FIPS_NAMES.get(state_fips)
    if name:
        return f"{state_fips} {name}"
    return state_fips


def summarize(rows, assigned, written, crs_info, tiger_stats, shp_path, businesses_path, businesses_hash, checks, out_path):
    by_county = {}
    for row in written:
        if row["match_status"] != "matched":
            continue
        item = by_county.setdefault(
            row["county_fips"],
            {
                "county_fips": row["county_fips"],
                "county_name": row["county_name"],
                "state_fips": row["state_fips"],
                "state_label": state_label(row["state_fips"]),
                "n_businesses": 0,
                "nyc_five_boroughs": row["county_fips"] in NYC_FIVE_BOROUGHS,
            },
        )
        item["n_businesses"] += 1
    counties = sorted(by_county.values(), key=lambda item: (-item["n_businesses"], item["county_fips"]))
    boroughs = []
    for fips, expected in sorted(NYC_FIVE_BOROUGHS.items()):
        found = by_county.get(fips)
        tiger_name = None if found is None else found["county_name"]
        boroughs.append(
            {
                "county_fips": fips,
                "expected_name": expected,
                "tiger_name": tiger_name,
                "n_businesses": 0 if found is None else found["n_businesses"],
                "name_matches_expected": tiger_name == expected if tiger_name else False,
            }
        )
    # If a borough had zero matches it may still have been a candidate. Fill the
    # TIGER name from a side table passed through tiger_stats when present.
    names_from_candidates = tiger_stats.get("candidate_names", {})
    for item in boroughs:
        if item["tiger_name"] is None and item["county_fips"] in names_from_candidates:
            item["tiger_name"] = names_from_candidates[item["county_fips"]]
            item["name_matches_expected"] = item["tiger_name"] == item["expected_name"]
    unmatched = [
        {"gmap_id": row["gmap_id"], "latitude": row["latitude"], "longitude": row["longitude"]}
        for row in assigned
        if row["match_status"] == "unmatched"
    ]
    ambiguous = [
        {
            "gmap_id": row["gmap_id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "candidates": row["candidates"],
        }
        for row in assigned
        if row["match_status"] == "ambiguous"
    ]
    invalid = [
        {"gmap_id": row["gmap_id"], "latitude": row["latitude"], "longitude": row["longitude"]}
        for row in assigned
        if row["match_status"] == "invalid_coordinates"
    ]
    counts = {
        "businesses_processed": len(rows),
        "matched": sum(1 for row in written if row["match_status"] == "matched"),
        "unmatched": len(unmatched),
        "ambiguous": len(ambiguous),
        "invalid_coordinates": len(invalid),
        "unique_counties": len(by_county),
        "output_rows": len(written),
        "businesses_in_nyc_five_boroughs": sum(item["n_businesses"] for item in boroughs),
        "businesses_outside_nyc_five_boroughs": sum(
            item["n_businesses"] for item in counties if not item["nyc_five_boroughs"]
        ),
    }
    observations = []
    outside_states = sorted({item["state_fips"] for item in counties if item["state_fips"] != "36"})
    if outside_states:
        observations.append(
            "Matched counties include state FIPS "
            + ", ".join(state_label(code) for code in outside_states)
            + ". The sampling box is not a New York State boundary, so these matches were kept."
        )
    outside_boroughs = [item for item in counties if not item["nyc_five_boroughs"]]
    if outside_boroughs:
        observations.append(
            f"{counts['businesses_outside_nyc_five_boroughs']} matched businesses are outside the five NYC borough counties."
        )
    if unmatched:
        observations.append(f"{len(unmatched)} businesses have coordinates covered by no county polygon.")
    if ambiguous:
        observations.append(f"{len(ambiguous)} businesses fall on a boundary covered by more than one county.")
    if invalid:
        observations.append(f"{len(invalid)} businesses have coordinates that are missing or outside the valid latitude/longitude range.")
    name_mismatches = [item for item in boroughs if item["tiger_name"] and not item["name_matches_expected"]]
    if name_mismatches:
        observations.append(
            "TIGER NAME differs from the expected borough name for "
            + ", ".join(item["county_fips"] for item in name_mismatches)
            + "."
        )
    missing_boroughs = [item["county_fips"] for item in boroughs if item["tiger_name"] is None]
    if missing_boroughs:
        observations.append(
            "These NYC borough FIPS codes were not present among candidate TIGER counties: "
            + ", ".join(missing_boroughs)
            + "."
        )
    if tiger_stats["invalid_geometries_repaired"]:
        observations.append(
            f"{tiger_stats['invalid_geometries_repaired']} candidate county geometries were invalid and passed through make_valid before matching."
        )
    return {
        "businesses_file": rel(businesses_path),
        "businesses_sha256": businesses_hash,
        "crosswalk_file": rel(out_path),
        "tiger": {
            "url": TIGER_URL,
            "zip_sha256": sha256(TIGER_ZIP),
            "shapefile": rel(shp_path),
            "n_county_records": tiger_stats["n_county_records"],
            "n_candidate_counties": tiger_stats["n_candidate_counties"],
            "candidate_state_fips": tiger_stats["candidate_state_fips"],
            "candidate_filter": (
                "A county is tested only when its shapefile bounding box intersects the "
                "transformed business extent, padded by 0.02 degrees. The pad is a candidate "
                "index, not an assignment. Assignment uses polygon covers and does not snap "
                "to the nearest county."
            ),
            "invalid_geometries_repaired": tiger_stats["invalid_geometries_repaired"],
            "records_skipped_bad_geoid": tiger_stats["records_skipped_bad_geoid"],
            "records_with_statefp_geoid_mismatch": tiger_stats["records_with_statefp_geoid_mismatch"],
        },
        "crs": crs_info,
        "counts": counts,
        "nyc_five_boroughs": boroughs,
        "counties_by_business_count": counties,
        "unmatched_businesses": unmatched,
        "ambiguous_businesses": ambiguous,
        "invalid_coordinate_businesses": invalid,
        "observations": observations,
        "validation": {"passed": all(item["passed"] for item in checks), "checks": checks},
    }


def main(state):
    businesses_path = ROOT / "data" / "processed" / f"businesses_{state}.jsonl"
    if not businesses_path.exists():
        raise SystemExit(f"Missing {businesses_path}")
    businesses_hash = sha256(businesses_path)
    rows, duplicate_ids = load_businesses(businesses_path)
    if len(rows) != 10000 or duplicate_ids or any(not isinstance(row["gmap_id"], str) or not row["gmap_id"] for row in rows):
        raise SystemExit(
            f"Refusing to write a crosswalk: expected 10,000 unique gmap_id values, "
            f"found {len(rows)} rows and {len(duplicate_ids)} duplicate ids."
        )

    download_tiger(TIGER_URL, TIGER_ZIP)
    shp_path = extract_tiger(TIGER_ZIP, TIGER_DIR)
    prj_path = shp_path.with_suffix(".prj")
    crs, epsg, equals_4269, reference = load_crs(prj_path)
    crs_info = {
        "business_coordinates_assumed_epsg": 4326,
        "business_coordinates_assumption": (
            "businesses_New_York.jsonl stores latitude and longitude and does not store a CRS. "
            "The script treats those numbers as EPSG:4326 longitude/latitude, then transforms "
            "them into the TIGER CRS before point-in-polygon."
        ),
        "tiger_crs_name": crs.name,
        "tiger_epsg": epsg,
        "tiger_equals_epsg_4269_ignoring_axis_order": equals_4269,
        "tiger_strict_equals_epsg_4269": bool(crs.equals(reference)),
        "tiger_axis_order": [axis.direction for axis in crs.axis_info],
        "epsg_4269_axis_order": [axis.direction for axis in reference.axis_info],
        "tiger_is_geographic": bool(crs.is_geographic),
        "transformer": "pyproj Transformer.from_crs(EPSG:4326, tiger CRS, always_xy=True)",
        "predicate": (
            "STRtree bbox candidates, then county_polygon.covers(point). "
            "covers includes the boundary and excludes holes. The STRtree predicate "
            "argument is not used: Shapely 2.1 evaluates it as predicate(query, tree geometry)."
        ),
    }
    if not crs.is_geographic or not equals_4269:
        raise SystemExit(
            f"TIGER CRS is {crs.name} (EPSG:{epsg}, geographic={crs.is_geographic}, "
            f"equals EPSG:4269 ignoring axis order={equals_4269}). Expected geographic NAD83 / EPSG:4269."
        )
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    valid_rows = [row for row in rows if valid_coordinate(row["latitude"], row["longitude"])]
    if not valid_rows:
        raise SystemExit("No valid coordinates in the business file.")
    sample_xy = [transformer.transform(row["longitude"], row["latitude"]) for row in valid_rows]
    minx = min(pair[0] for pair in sample_xy)
    maxx = max(pair[0] for pair in sample_xy)
    miny = min(pair[1] for pair in sample_xy)
    maxy = max(pair[1] for pair in sample_xy)
    candidates, tiger_stats = load_candidate_counties(shp_path, (minx, miny, maxx, maxy))
    tiger_stats["candidate_names"] = {item["county_fips"]: item["county_name"] for item in candidates}
    tiger_stats["candidate_state_fips"] = sorted({item["state_fips"] for item in candidates})
    assigned, transformed = assign(rows, transformer, candidates)
    written = [{key: row[key] for key in CROSSWALK_FIELDS} for row in assigned]

    out_path = ROOT / "data" / "processed" / f"county_crosswalk_{state}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in written:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    checks = build_checks(
        rows,
        duplicate_ids,
        assigned,
        transformed,
        crs_info,
        businesses_path,
        businesses_hash,
        written,
    )
    # Re-read the file so the FIPS-type check sees JSON strings, not the in-memory rows only.
    reread = []
    with out_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            reread.append(item)
    check(
        checks,
        "reread_row_count",
        len(reread) == len(written) == 10000,
        f"{len(reread)} lines in {rel(out_path)}",
    )
    reread_ok = reread == written
    check(checks, "reread_matches_written_rows", reread_ok, "JSONL round-trip preserves FIPS strings and nulls")
    summary = summarize(
        rows,
        assigned,
        written,
        crs_info,
        tiger_stats,
        shp_path,
        businesses_path,
        businesses_hash,
        checks,
        out_path,
    )
    summary_path = ROOT / "results" / "tables" / f"county_crosswalk_{state}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": summary["counts"], "validation_passed": summary["validation"]["passed"]}, indent=2))
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}: {item['detail']}")
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")
    if not summary["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="New_York")
    args = parser.parse_args()
    main(args.state)
