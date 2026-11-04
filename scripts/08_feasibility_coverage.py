#!/usr/bin/env python3
"""Geographic and image-coverage feasibility checks on the processed NYC sample.

Reads data/processed/businesses_New_York.jsonl and reviews_New_York.jsonl.
Does not download source data, rerun preprocessing, or fit models.
Writes aggregate tables and figures only. Review text, business names, and
image URLs are not saved.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
LAT0, LAT1 = 40.48, 40.93
LON0, LON1 = -74.26, -73.68
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(40.7))

# Integer grids over the sampling box. Degree steps differ so cells are
# closer to equal ground distance. They are not census tracts or neighborhoods.
GRIDS = (
    {
        "name": "25x25",
        "rows": 25,
        "cols": 25,
        "min_businesses_for_rate": 20,
        "figure_tag": "25x25",
    },
    {
        "name": "9x10",
        "rows": 9,
        "cols": 10,
        "min_businesses_for_rate": 30,
        "figure_tag": "9x10",
    },
)


def load_image_urls():
    spec = importlib.util.spec_from_file_location("preprocess", ROOT / "scripts" / "03_preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.image_urls


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels_of(raw):
    if not isinstance(raw, list):
        return []
    seen = set()
    labels = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in seen:
            seen.add(text)
            labels.append(text)
    return labels


def percentile(values, q):
    xs = sorted(values)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    weight = pos - lo
    return xs[lo] * (1 - weight) + xs[hi] * weight


def gini(values):
    xs = sorted(float(v) for v in values)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total == 0:
        return None
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * weighted) / (n * total) - (n + 1) / n


@dataclass
class Business:
    lat: float
    lon: float
    categories: list
    listed_reviews: int | None
    retained: int = 0
    image_reviews: int = 0
    url_occurrences: int = 0
    urls: set = field(default_factory=set)


@dataclass
class GridSpec:
    name: str
    rows: int
    cols: int
    min_businesses_for_rate: int

    @property
    def lat_step(self):
        return (LAT1 - LAT0) / self.rows

    @property
    def lon_step(self):
        return (LON1 - LON0) / self.cols

    def index(self, lat, lon):
        if not (LAT0 <= lat <= LAT1 and LON0 <= lon <= LON1):
            return None
        row = min(self.rows - 1, max(0, int((lat - LAT0) / self.lat_step)))
        col = min(self.cols - 1, max(0, int((lon - LON0) / self.lon_step)))
        return row, col

    def bounds(self, row, col):
        return (
            LAT0 + row * self.lat_step,
            LAT0 + (row + 1) * self.lat_step,
            LON0 + col * self.lon_step,
            LON0 + (col + 1) * self.lon_step,
        )


def load_businesses(path):
    businesses = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            lat, lon = item.get("latitude"), item.get("longitude")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                raise SystemExit(f"Business missing coordinates in {path}")
            listed = item.get("num_of_reviews")
            if not isinstance(listed, (int, float)):
                listed = None
            else:
                listed = int(listed)
            gmap_id = item.get("gmap_id")
            if not gmap_id or gmap_id in businesses:
                raise SystemExit("Expected one unique gmap_id per business row")
            businesses[gmap_id] = Business(float(lat), float(lon), labels_of(item.get("category")), listed)
    return businesses


def accumulate_reviews(path, businesses, image_urls):
    unknown = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            business = businesses.get(item.get("gmap_id"))
            if business is None:
                unknown += 1
                continue
            business.retained += 1
            urls = image_urls(item.get("pics"))
            if not urls:
                continue
            business.image_reviews += 1
            business.url_occurrences += len(urls)
            business.urls.update(urls)
    return unknown


def new_cell():
    return {
        "n_businesses": 0,
        "n_retained_reviews": 0,
        "n_image_bearing_reviews": 0,
        "n_businesses_with_image": 0,
        "image_url_occurrences": 0,
        "urls": set(),
    }


def fill_grid(spec, businesses):
    cells = defaultdict(new_cell)
    outside = 0
    for business in businesses.values():
        key = spec.index(business.lat, business.lon)
        if key is None:
            outside += 1
            continue
        cell = cells[key]
        cell["n_businesses"] += 1
        cell["n_retained_reviews"] += business.retained
        cell["n_image_bearing_reviews"] += business.image_reviews
        cell["image_url_occurrences"] += business.url_occurrences
        if business.image_reviews:
            cell["n_businesses_with_image"] += 1
        cell["urls"].update(business.urls)
    return cells, outside


def cell_rows(spec, cells):
    rows = []
    for row in range(spec.rows):
        for col in range(spec.cols):
            cell = cells.get((row, col), new_cell())
            south, north, west, east = spec.bounds(row, col)
            n = cell["n_businesses"]
            with_image = cell["n_businesses_with_image"]
            rows.append(
                {
                    "cell_id": f"r{row:02d}c{col:02d}",
                    "row_south_to_north": row,
                    "col_west_to_east": col,
                    "lat_south": round(south, 6),
                    "lat_north": round(north, 6),
                    "lon_west": round(west, 6),
                    "lon_east": round(east, 6),
                    "n_businesses": n,
                    "n_retained_reviews": cell["n_retained_reviews"],
                    "n_image_bearing_reviews": cell["n_image_bearing_reviews"],
                    "n_businesses_with_image": with_image,
                    "pct_businesses_with_image": round(100 * with_image / n, 4) if n else "",
                    "image_url_occurrences": cell["image_url_occurrences"],
                    "unique_image_urls": len(cell["urls"]),
                }
            )
    return rows


def grid_summary(spec, cells):
    counts = []
    retained = []
    image_reviews = []
    rates = []
    occupied_counts = []
    for row in range(spec.rows):
        for col in range(spec.cols):
            cell = cells.get((row, col))
            n = 0 if cell is None else cell["n_businesses"]
            counts.append(n)
            retained.append(0 if cell is None else cell["n_retained_reviews"])
            image_reviews.append(0 if cell is None else cell["n_image_bearing_reviews"])
            if n:
                occupied_counts.append(n)
            if n >= spec.min_businesses_for_rate:
                rates.append(100 * cell["n_businesses_with_image"] / n)
    occupied = [v for v in counts if v]
    densest = max(
        ((n, row, col) for (row, col), cell in cells.items() if (n := cell["n_businesses"])),
        default=(0, None, None),
    )
    top_share = None
    if occupied_counts:
        ranked = sorted(occupied_counts, reverse=True)
        k = max(1, math.ceil(0.1 * len(ranked)))
        top_share = 100 * sum(ranked[:k]) / sum(ranked)
    center = None
    if densest[1] is not None:
        south, north, west, east = spec.bounds(densest[1], densest[2])
        center = ((south + north) / 2, (west + east) / 2)
    return {
        "name": spec.name,
        "rows": spec.rows,
        "cols": spec.cols,
        "n_cells": spec.rows * spec.cols,
        "lat_step_degrees": round(spec.lat_step, 4),
        "lon_step_degrees": round(spec.lon_step, 4),
        "approx_km_lat": round(spec.lat_step * KM_PER_DEG_LAT, 2),
        "approx_km_lon": round(spec.lon_step * KM_PER_DEG_LON, 2),
        "min_businesses_for_rate": spec.min_businesses_for_rate,
        "occupied_cells": len(occupied),
        "empty_cells": spec.rows * spec.cols - len(occupied),
        "empty_cell_pct": round(100 * (spec.rows * spec.cols - len(occupied)) / (spec.rows * spec.cols), 2),
        "business_count_min_occupied": min(occupied) if occupied else None,
        "business_count_median_occupied": statistics.median(occupied) if occupied else None,
        "business_count_p90_occupied": percentile(occupied, 0.9),
        "business_count_max": max(counts) if counts else None,
        "retained_reviews_max": max(retained) if retained else None,
        "image_bearing_reviews_max": max(image_reviews) if image_reviews else None,
        "gini_all_cells": None if gini(counts) is None else round(gini(counts), 3),
        "gini_occupied_cells": None if gini(occupied) is None else round(gini(occupied), 3),
        "top_decile_occupied_cells_share_of_businesses_pct": None if top_share is None else round(top_share, 2),
        "densest_cell_businesses": densest[0],
        "densest_cell_center_lat": None if center is None else center[0],
        "densest_cell_center_lon": None if center is None else center[1],
        "rate_cells": len(rates),
        "rate_min_pct": None if not rates else round(min(rates), 2),
        "rate_median_pct": None if not rates else round(statistics.median(rates), 2),
        "rate_max_pct": None if not rates else round(max(rates), 2),
        "matrices": {
            "businesses": counts_matrix(spec, cells, "n_businesses"),
            "retained": counts_matrix(spec, cells, "n_retained_reviews"),
            "image_reviews": counts_matrix(spec, cells, "n_image_bearing_reviews"),
            "rate": rate_matrix(spec, cells),
        },
    }


def counts_matrix(spec, cells, key):
    matrix = np.zeros((spec.rows, spec.cols), dtype=float)
    for (row, col), cell in cells.items():
        matrix[row, col] = cell[key]
    return matrix


def rate_matrix(spec, cells):
    matrix = np.full((spec.rows, spec.cols), np.nan)
    for (row, col), cell in cells.items():
        n = cell["n_businesses"]
        if n >= spec.min_businesses_for_rate:
            matrix[row, col] = 100 * cell["n_businesses_with_image"] / n
    return matrix


def category_rows(businesses, min_businesses):
    stats = defaultdict(
        lambda: {
            "n_businesses": 0,
            "n_businesses_with_image": 0,
            "n_retained_reviews": 0,
            "n_image_bearing_reviews": 0,
            "image_url_occurrences": 0,
            "urls": set(),
            "n_businesses_if_first_label_only": 0,
            "n_first_label_with_image": 0,
        }
    )
    for business in businesses.values():
        first = business.categories[0] if business.categories else None
        for label in business.categories:
            row = stats[label]
            row["n_businesses"] += 1
            row["n_retained_reviews"] += business.retained
            row["n_image_bearing_reviews"] += business.image_reviews
            row["image_url_occurrences"] += business.url_occurrences
            row["urls"].update(business.urls)
            if business.image_reviews:
                row["n_businesses_with_image"] += 1
        if first is not None:
            stats[first]["n_businesses_if_first_label_only"] += 1
            if business.image_reviews:
                stats[first]["n_first_label_with_image"] += 1
    rows = []
    for label, row in stats.items():
        if row["n_businesses"] < min_businesses:
            continue
        n = row["n_businesses"]
        reviews = row["n_retained_reviews"]
        first_n = row["n_businesses_if_first_label_only"]
        rows.append(
            {
                "category": label,
                "n_businesses": n,
                "n_businesses_with_image": row["n_businesses_with_image"],
                "pct_businesses_with_image": round(100 * row["n_businesses_with_image"] / n, 4),
                "mean_image_bearing_reviews_per_business": round(row["n_image_bearing_reviews"] / n, 4),
                "n_retained_reviews": reviews,
                "n_image_bearing_reviews": row["n_image_bearing_reviews"],
                "pct_retained_reviews_with_image": round(100 * row["n_image_bearing_reviews"] / reviews, 4) if reviews else "",
                "image_url_occurrences": row["image_url_occurrences"],
                "unique_image_urls": len(row["urls"]),
                "n_businesses_if_first_label_only": first_n,
                "pct_businesses_with_image_if_first_label_only": round(100 * row["n_first_label_with_image"] / first_n, 4) if first_n else "",
            }
        )
    rows.sort(key=lambda item: (-item["n_businesses"], item["category"]))
    omitted = sum(1 for row in stats.values() if row["n_businesses"] < min_businesses)
    return rows, len(stats), omitted


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(businesses):
    counts = defaultdict(int)
    for business in businesses.values():
        counts[business.image_reviews] += 1
    total = len(businesses)
    rows = []
    for k in range(0, max(counts) + 1):
        n = counts[k]
        rows.append(
            {
                "image_bearing_retained_reviews": k,
                "n_businesses": n,
                "pct_of_sampled_businesses": round(100 * n / total, 4),
            }
        )
    return rows


def style_axes():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def extent():
    return (LON0, LON1, LAT0, LAT1)


def draw_count(ax, matrix, title, colorbar_label):
    positive = matrix[matrix > 0]
    ax.set_facecolor("#f3f3f3")
    if positive.size == 0:
        ax.set_title(title)
        return
    masked = np.ma.masked_where(matrix <= 0, matrix)
    image = ax.imshow(
        masked,
        origin="lower",
        extent=extent(),
        cmap="Blues",
        norm=LogNorm(vmin=1, vmax=max(positive.max(), 1)),
        interpolation="nearest",
        aspect="equal",
    )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.tick_params(labelsize=8)


def draw_rate(ax, matrix, title, colorbar_label):
    ax.set_facecolor("#e6e6e6")
    finite = matrix[np.isfinite(matrix)]
    vmax = 5 if finite.size == 0 else max(5, math.ceil(float(finite.max()) / 5) * 5)
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        origin="lower",
        extent=extent(),
        cmap="cividis",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.tick_params(labelsize=8)


def save_scatter(path, businesses):
    without = [(b.lon, b.lat) for b in businesses.values() if b.image_reviews == 0]
    with_image = [(b.lon, b.lat) for b in businesses.values() if b.image_reviews > 0]
    fig, ax = plt.subplots(figsize=(7.2, 8.0), dpi=160)
    ax.scatter(
        [p[0] for p in without],
        [p[1] for p in without],
        s=7,
        c="#9aa0a6",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        label=f"No image-bearing retained review ({len(without):,})",
    )
    ax.scatter(
        [p[0] for p in with_image],
        [p[1] for p in with_image],
        s=8,
        c="#0b6e4f",
        alpha=0.55,
        linewidths=0,
        rasterized=True,
        label=f"At least one image-bearing retained review ({len(with_image):,})",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title("Sampled business coordinates in the NYC sampling box")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.005,
        "One point per sampled business. Colors mark image-URL presence in retained reviews, not neighborhoods. Source: processed NYC sample.",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def save_count_panels(path, summary, caption):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), dpi=160)
    draw_count(axes[0], summary["matrices"]["businesses"], "Sampled businesses", "Businesses per cell (log scale)")
    draw_count(axes[1], summary["matrices"]["retained"], "Retained reviews", "Retained reviews per cell (log scale)")
    draw_count(
        axes[2],
        summary["matrices"]["image_reviews"],
        "Image-bearing retained reviews",
        "Image-bearing reviews per cell (log scale)",
    )
    fig.suptitle(f"Exploratory {summary['name']} grid over the sampling box", fontsize=13)
    fig.text(0.01, 0.005, caption, fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def save_rate(path, summary, overall_pct):
    fig, ax = plt.subplots(figsize=(7.2, 8.0), dpi=160)
    minimum = summary["min_businesses_for_rate"]
    draw_rate(
        ax,
        summary["matrices"]["rate"],
        f"Businesses with an image-bearing retained review\nExploratory {summary['name']} grid, cells with at least {minimum} businesses",
        "Percent of businesses in the cell",
    )
    fig.text(
        0.01,
        0.005,
        f"Gray cells have fewer than {minimum} sampled businesses and are not given a rate. "
        f"Sample-wide, {overall_pct:.1f}% of businesses have an image-bearing retained review. "
        "This is not a neighborhood map. Source: processed NYC sample.",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def save_histogram(path, rows, n_businesses, n_with_image):
    ks = [row["image_bearing_retained_reviews"] for row in rows]
    ns = [row["n_businesses"] for row in rows]
    positive = [(k, n) for k, n in zip(ks, ns) if k > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=160)
    axes[0].bar(ks, ns, color="#24527a", width=0.85)
    axes[0].set_xlabel("Image-bearing retained reviews per business")
    axes[0].set_ylabel("Sampled businesses")
    axes[0].set_title(f"All sampled businesses (n={n_businesses:,})")
    if positive:
        axes[1].bar([k for k, _ in positive], [n for _, n in positive], color="#0b6e4f", width=0.85)
    axes[1].set_xlabel("Image-bearing retained reviews per business")
    axes[1].set_ylabel("Sampled businesses")
    axes[1].set_title(f"Businesses with at least one (n={n_with_image:,})")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Distribution of image-bearing retained reviews", fontsize=13)
    fig.text(
        0.01,
        0.005,
        "Counts are retained reviews that contain an image URL. The preprocessing cap is 20 retained reviews per business. Source: processed NYC sample.",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def save_categories(path, rows):
    chosen = [row for row in rows if row["n_businesses"] >= 100]
    chosen = sorted(chosen, key=lambda row: row["n_businesses"], reverse=True)[:18]
    chosen = sorted(chosen, key=lambda row: row["pct_businesses_with_image"])
    fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=160)
    if chosen:
        y = np.arange(len(chosen))
        ax.barh(y, [row["pct_businesses_with_image"] for row in chosen], color="#24527a")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{row['category']} (n={row['n_businesses']:,})" for row in chosen], fontsize=8)
    ax.set_xlabel("Percent of businesses with an image-bearing retained review")
    ax.set_ylabel("Category label")
    ax.set_title("Image coverage for the largest multi-label categories")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.005,
        "A business is counted in every category it carries. Percentages overlap and do not sum to 100. Denominator is sampled businesses with that label. Source: processed NYC sample.",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fmt(value, digits=1):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def write_brief(path, summary):
    fine = summary["grids"]["25x25"]
    coarse = summary["grids"]["9x10"]
    spread = summary["category_spread_n_at_least_100"]
    unstable = summary["first_label_instability"]
    plotted = summary["plotted_category_spread"]
    review_match = (
        "Every retained review matches a sampled business."
        if summary["reviews_for_unknown_business"] == 0
        else f"{summary['reviews_for_unknown_business']:,} review rows did not match a sampled business."
    )
    text = f"""# Feasibility notes for 6 November

These checks describe the processed New York sample already on disk. They do not re-download the UCSD Google Local files, rerun sampling, or fit embeddings or clustering models.

## Data frame

The business file contains {summary["n_businesses"]:,} sampled businesses, all with coordinates. The review file contains {summary["n_retained_reviews"]:,} retained reviews. {review_match} Preprocessing, as recorded in the existing coverage file, drew a uniform sample of 10,000 businesses from the NYC bounding box with seed 42, then kept at most 20 reviews per business with a reservoir. Eligible reviews for that reservoir were reviews with text or an image URL.

{summary["businesses_with_retained_reviews"]:,} businesses have at least one retained review. {summary["businesses_without_retained_reviews"]:,} have none. Those {summary["businesses_without_retained_reviews"]:,} still count in business-level denominators and in the coordinate plot. They add nothing to review or image-URL counts.

The sampling window is latitude 40.48–40.93 and longitude −74.26 to −73.68. It is the box already used in preprocessing. It is not the city boundary. Sampled coordinates span latitude {summary["lat_min"]:.3f}–{summary["lat_max"]:.3f} and longitude {summary["lon_min"]:.3f}–{summary["lon_max"]:.3f}.

## Geographic coverage

No census-tract, borough, or neighborhood boundary file is in the repository. The units below are exploratory grids aligned to the sampling box. A grid cell is not a neighborhood.

The finer grid has 25 rows and 25 columns. Each cell is 0.018° of latitude by 0.0232° of longitude, about {fine["approx_km_lat"]:.1f} km by {fine["approx_km_lon"]:.1f} km at 40.7°N. The coarser check has 9 rows and 10 columns, 0.050° by 0.058°, about {coarse["approx_km_lat"]:.1f} km by {coarse["approx_km_lon"]:.1f} km. Degree steps differ because a degree of longitude is shorter than a degree of latitude at this latitude. Cell land area is still unequal: the box contains water and some land outside New York City.

On the 25×25 grid, {fine["occupied_cells"]} of {fine["n_cells"]} cells contain a sampled business and {fine["empty_cells"]} ({fmt(fine["empty_cell_pct"])}%) are empty. Among occupied cells the median count is {fmt(fine["business_count_median_occupied"], 0)} and the maximum is {fine["business_count_max"]:,}, in the cell centered near latitude {fine["densest_cell_center_lat"]:.3f}, longitude {fine["densest_cell_center_lon"]:.3f}. The densest tenth of occupied cells contains {fmt(fine["top_decile_occupied_cells_share_of_businesses_pct"])}% of the businesses that fall in occupied cells. The Gini coefficient of business counts is {fmt(fine["gini_all_cells"], 2)} across all cells and {fmt(fine["gini_occupied_cells"], 2)} across occupied cells.

The 9×10 grid is less sparse and still uneven: {coarse["empty_cells"]} of {coarse["n_cells"]} cells are empty ({fmt(coarse["empty_cell_pct"])}%), the occupied median is {fmt(coarse["business_count_median_occupied"], 0)}, the maximum is {coarse["business_count_max"]:,}, and the all-cell Gini coefficient is {fmt(coarse["gini_all_cells"], 2)}. Empty cells are an expected mix of water and land with no sampled business. They are not measurements of neighborhoods with zero activity.

Businesses were sampled uniformly from listed businesses, not uniformly across land. Commercial corridors with many listings contribute more points than residential blocks. That is spatial sampling bias relative to a map of places. Half-box shares, which only describe the point cloud, put {fmt(summary["share_south_half_pct"])}% of businesses in the southern half of the sampling box and {fmt(summary["share_west_half_pct"])}% in the western half.

Retained review counts are also uneven, and they are capped. {summary["businesses_at_review_cap"]:,} businesses are at the 20-review cap. The median `num_of_reviews` stored on the business record is {fmt(summary["listed_reviews_median"], 0)}, and {fmt(summary["listed_reviews_above_cap_pct"])}% of businesses with a listed count are above 20. The retained file is a capped subset of eligible reviews, not the full Google review history. Image-bearing review counts use that same cap.

## Image coverage and bias

Image URL occurrences, unique URLs, reviews with images, and businesses with images are separate counts:

- Image URL occurrences in retained reviews: {summary["image_url_occurrences"]:,}
- Unique image URLs: {summary["unique_image_urls"]:,}
- Retained reviews with at least one image URL: {summary["image_bearing_reviews"]:,} ({fmt(summary["pct_retained_reviews_with_image"])}% of retained reviews)
- Businesses with at least one image-bearing retained review: {summary["businesses_with_image"]:,} ({fmt(summary["pct_businesses_with_image"])}% of the {summary["n_businesses"]:,} sampled businesses)

Occurrences and unique URLs differ by {summary["image_url_occurrences"] - summary["unique_image_urls"]:,}, so repeated use of the same URL is uncommon in the retained sample. A business can clear the business-level threshold with a single image-bearing review. That is why {fmt(summary["pct_businesses_with_image"])}% of businesses have an image while {fmt(summary["pct_retained_reviews_with_image"])}% of retained reviews do.

The business-level percentage uses every sampled business as the denominator, including businesses with no retained review. Among businesses that have at least one retained review, the share is {fmt(summary["pct_businesses_with_image_given_review"])}%. The review-level percentage uses retained reviews as the denominator and answers a different question.

Per sampled business, the number of image-bearing retained reviews has median {fmt(summary["image_reviews_per_business_median"], 0)}, mean {fmt(summary["image_reviews_per_business_mean"], 2)}, and 90th percentile {fmt(summary["image_reviews_per_business_p90"], 0)}. {fmt(summary["pct_businesses_with_zero_image_reviews"])}% have zero. Among the businesses with at least one, the median is {fmt(summary["image_reviews_per_business_median_if_any"], 0)} and the maximum is {summary["image_reviews_per_business_max"]}. The cap of 20 retained reviews bounds that maximum.

Counts of image-bearing reviews follow the same concentrated pattern as the business counts. The share of businesses with an image does not. On the 25×25 grid, {fine["rate_cells"]} cells have at least {fine["min_businesses_for_rate"]} businesses. Their business-level image share ranges from {fmt(fine["rate_min_pct"])}% to {fmt(fine["rate_max_pct"])}% (median {fmt(fine["rate_median_pct"])}%). On the 9×10 grid, {coarse["rate_cells"]} cells have at least {coarse["min_businesses_for_rate"]} businesses, and the share ranges from {fmt(coarse["rate_min_pct"])}% to {fmt(coarse["rate_max_pct"])}% (median {fmt(coarse["rate_median_pct"])}%). Gray cells on the rate maps are below those cutoffs. The sample-wide business rate is {fmt(summary["pct_businesses_with_image"])}%, so the geographic range is real but much smaller than the range in raw counts.

Categories are multi-label. {summary["businesses_with_multiple_categories"]:,} businesses have two or more labels, {summary["businesses_with_no_category"]} have none, and there are {summary["category_assignments"]:,} label assignments across {summary["n_categories"]:,} distinct labels (mean {fmt(summary["mean_labels_per_business"], 2)} labels per business). A business enters the denominator of every category it carries. Rows overlap and cannot be added into a partition of the 10,000 businesses. The category table keeps labels with at least 20 sampled businesses ({summary["category_rows_written"]:,} labels; {summary["categories_omitted_below_20"]:,} rarer labels are omitted). The first-label columns use the first string in the stored list. That order is not a verified primary category.

Among labels with at least 100 sampled businesses ({spread["n_categories"]} labels), the business-level image share ranges from {fmt(spread["min_pct"])}% for {spread["min_category"]} (n={spread["min_n"]:,}) to {fmt(spread["max_pct"])}% for {spread["max_category"]} (n={spread["max_n"]:,}). The category figure is the 18 most common labels only, ordered by this percentage. Those labels run from {fmt(plotted["min_pct"])}% for {plotted["min_category"]} (n={plotted["min_n"]:,}) to {fmt(plotted["max_pct"])}% for {plotted["max_category"]} (n={plotted["max_n"]:,}). First-label percentages are unstable when few businesses carry the label first. {unstable["category"]} is {fmt(unstable["pct"])}% across {unstable["n_businesses"]:,} businesses, but {fmt(unstable["first_pct"])}% among the {unstable["first_n"]:,} businesses for which it is the first stored label. The multi-label denominator is the one used here. These gaps are differences in image-URL presence. They are not evidence that image features improve a representation of neighborhoods.

## Limitations

The box and the grids do not follow official boundaries, and empty cells mix water with unsampled land. Uniform sampling of businesses over-represents places with many listings. Review retention is capped at 20 eligible reviews, so businesses with long review histories are only partly observed. Reviews with neither text nor an image URL were dropped before the reservoir, so the image rate among retained reviews is not the image rate among every stored review. This count treats a non-empty image URL as coverage. It does not open the file. A separate 200-URL download, already on disk, decoded 194 images and found 6 HTTP 404 responses, so URL presence is close to file availability in that small check and is not a census of pixels. No embedding or clustering model was trained. These results do not show that adding image features improves neighborhood representation.

## Files

- `results/figures/sampled_business_coordinates.png`
- `results/figures/exploratory_grid_25x25_counts.png`
- `results/figures/exploratory_grid_25x25_image_business_share.png`
- `results/figures/exploratory_grid_9x10_counts.png`
- `results/figures/exploratory_grid_9x10_image_business_share.png`
- `results/figures/image_bearing_reviews_per_business.png`
- `results/figures/image_coverage_by_category.png`
- `results/tables/exploratory_grid_25x25_New_York.csv`
- `results/tables/exploratory_grid_9x10_New_York.csv`
- `results/tables/category_image_coverage_New_York.csv`
- `results/tables/image_review_distribution_New_York.csv`
- `results/tables/feasibility_summary_New_York.json`
"""
    path.write_text(text, encoding="utf-8")


def public_grid(summary):
    return {key: value for key, value in summary.items() if key != "matrices"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--businesses", type=Path, default=ROOT / "data" / "processed" / "businesses_New_York.jsonl")
    parser.add_argument("--reviews", type=Path, default=ROOT / "data" / "processed" / "reviews_New_York.jsonl")
    parser.add_argument("--coverage", type=Path, default=ROOT / "results" / "tables" / "coverage_New_York.json")
    args = parser.parse_args()
    for path in (args.businesses, args.reviews):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    image_urls = load_image_urls()
    businesses = load_businesses(args.businesses)
    unknown_reviews = accumulate_reviews(args.reviews, businesses, image_urls)

    n_businesses = len(businesses)
    retained = [b.retained for b in businesses.values()]
    image_counts = [b.image_reviews for b in businesses.values()]
    n_reviews = sum(retained)
    n_image_reviews = sum(image_counts)
    n_with_image = sum(1 for value in image_counts if value)
    n_with_reviews = sum(1 for value in retained if value)
    url_occurrences = sum(b.url_occurrences for b in businesses.values())
    unique_urls = set()
    for business in businesses.values():
        unique_urls.update(business.urls)
    label_counts = [len(b.categories) for b in businesses.values()]
    listed = [b.listed_reviews for b in businesses.values() if b.listed_reviews is not None]
    lats = [b.lat for b in businesses.values()]
    lons = [b.lon for b in businesses.values()]
    positive = [value for value in image_counts if value]

    grids = []
    grid_tables = {}
    outside = {}
    for raw in GRIDS:
        spec = GridSpec(raw["name"], raw["rows"], raw["cols"], raw["min_businesses_for_rate"])
        cells, n_outside = fill_grid(spec, businesses)
        outside[spec.name] = n_outside
        grid_tables[spec.name] = cell_rows(spec, cells)
        grids.append(grid_summary(spec, cells))

    category_table, n_categories, omitted = category_rows(businesses, 20)
    large = [row for row in category_table if row["n_businesses"] >= 100]
    if not large:
        raise SystemExit("No category reached 100 businesses; the brief spread is undefined")
    low = min(large, key=lambda row: (row["pct_businesses_with_image"], row["category"]))
    high = max(large, key=lambda row: (row["pct_businesses_with_image"], row["category"]))
    unstable_pool = [
        row
        for row in large
        if 0 < row["n_businesses_if_first_label_only"] < 30
        and row["pct_businesses_with_image_if_first_label_only"] != ""
    ]
    plotted = sorted(large, key=lambda row: row["n_businesses"], reverse=True)[:18]
    plot_low = min(plotted, key=lambda row: (row["pct_businesses_with_image"], row["category"]))
    plot_high = max(plotted, key=lambda row: (row["pct_businesses_with_image"], row["category"]))
    unstable_row = max(
        unstable_pool,
        key=lambda row: abs(row["pct_businesses_with_image"] - row["pct_businesses_with_image_if_first_label_only"]),
    )
    dist_rows = distribution_rows(businesses)

    coverage_check = None
    if args.coverage.exists():
        recorded = json.loads(args.coverage.read_text(encoding="utf-8"))
        coverage_check = {
            "selected_businesses_match": recorded.get("selected_businesses") == n_businesses,
            "kept_reviews_match": recorded.get("kept_reviews") == n_reviews,
            "kept_with_pics_match": recorded.get("kept_with_pics") == n_image_reviews,
            "kept_image_urls_match": recorded.get("kept_image_urls") == url_occurrences,
            "unique_kept_image_urls_match": recorded.get("unique_kept_image_urls") == len(unique_urls),
            "businesses_with_kept_reviews_match": recorded.get("businesses_with_kept_reviews") == n_with_reviews,
        }
        if not all(coverage_check.values()):
            raise SystemExit(f"Processed files do not match {args.coverage}: {coverage_check}")

    summary = {
        "n_businesses": n_businesses,
        "n_retained_reviews": n_reviews,
        "reviews_for_unknown_business": unknown_reviews,
        "businesses_with_retained_reviews": n_with_reviews,
        "businesses_without_retained_reviews": n_businesses - n_with_reviews,
        "businesses_at_review_cap": sum(1 for value in retained if value >= 20),
        "listed_reviews_median": statistics.median(listed) if listed else None,
        "listed_reviews_above_cap_pct": 100 * sum(1 for value in listed if value > 20) / len(listed) if listed else None,
        "listed_reviews_missing": n_businesses - len(listed),
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lon_min": min(lons),
        "lon_max": max(lons),
        "share_south_half_pct": 100 * sum(1 for lat in lats if lat < (LAT0 + LAT1) / 2) / n_businesses,
        "share_west_half_pct": 100 * sum(1 for lon in lons if lon < (LON0 + LON1) / 2) / n_businesses,
        "image_url_occurrences": url_occurrences,
        "unique_image_urls": len(unique_urls),
        "image_bearing_reviews": n_image_reviews,
        "pct_retained_reviews_with_image": 100 * n_image_reviews / n_reviews if n_reviews else None,
        "businesses_with_image": n_with_image,
        "pct_businesses_with_image": 100 * n_with_image / n_businesses,
        "pct_businesses_with_image_given_review": 100 * n_with_image / n_with_reviews if n_with_reviews else None,
        "image_reviews_per_business_median": statistics.median(image_counts),
        "image_reviews_per_business_mean": statistics.fmean(image_counts),
        "image_reviews_per_business_p90": percentile(image_counts, 0.9),
        "image_reviews_per_business_max": max(image_counts),
        "image_reviews_per_business_median_if_any": statistics.median(positive) if positive else None,
        "pct_businesses_with_zero_image_reviews": 100 * (n_businesses - n_with_image) / n_businesses,
        "businesses_with_multiple_categories": sum(1 for n in label_counts if n > 1),
        "businesses_with_no_category": sum(1 for n in label_counts if n == 0),
        "category_assignments": sum(label_counts),
        "n_categories": n_categories,
        "mean_labels_per_business": statistics.fmean(label_counts),
        "category_rows_written": len(category_table),
        "categories_omitted_below_20": omitted,
        "plotted_category_spread": {
            "min_category": plot_low["category"],
            "min_pct": plot_low["pct_businesses_with_image"],
            "min_n": plot_low["n_businesses"],
            "max_category": plot_high["category"],
            "max_pct": plot_high["pct_businesses_with_image"],
            "max_n": plot_high["n_businesses"],
        },
        "first_label_instability": {
            "category": unstable_row["category"],
            "n_businesses": unstable_row["n_businesses"],
            "pct": unstable_row["pct_businesses_with_image"],
            "first_n": unstable_row["n_businesses_if_first_label_only"],
            "first_pct": unstable_row["pct_businesses_with_image_if_first_label_only"],
        },
        "category_spread_n_at_least_100": {
            "n_categories": len(large),
            "min_category": low["category"],
            "min_pct": low["pct_businesses_with_image"],
            "min_n": low["n_businesses"],
            "max_category": high["category"],
            "max_pct": high["pct_businesses_with_image"],
            "max_n": high["n_businesses"],
        },
        "businesses_outside_sampling_box": outside,
        "inputs": {
            "businesses": str(args.businesses.relative_to(ROOT)),
            "reviews": str(args.reviews.relative_to(ROOT)),
            "businesses_sha256": sha256(args.businesses),
            "reviews_sha256": sha256(args.reviews),
        },
        "coverage_reconciliation": coverage_check,
        "geographic_unit": "Exploratory grids over the preprocessing bounding box. Not census tracts, boroughs, or neighborhoods.",
        "grids": {item["name"]: item for item in grids},
    }

    figures = ROOT / "results" / "figures"
    tables = ROOT / "results" / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    style_axes()
    save_scatter(figures / "sampled_business_coordinates.png", businesses)
    by_name = {item["name"]: item for item in grids}
    caption_25 = (
        "Gray cells have no sampled business. Cells are about 2.0 km by 2.0 km and are not neighborhoods. "
        "Counts use a log color scale. Source: processed NYC sample."
    )
    caption_9 = (
        "Gray cells have no sampled business. Cells are about 5.6 km by 4.9 km and are not neighborhoods. "
        "Counts use a log color scale. Source: processed NYC sample."
    )
    save_count_panels(figures / "exploratory_grid_25x25_counts.png", by_name["25x25"], caption_25)
    save_count_panels(figures / "exploratory_grid_9x10_counts.png", by_name["9x10"], caption_9)
    overall_image_pct = 100 * n_with_image / n_businesses
    save_rate(figures / "exploratory_grid_25x25_image_business_share.png", by_name["25x25"], overall_image_pct)
    save_rate(figures / "exploratory_grid_9x10_image_business_share.png", by_name["9x10"], overall_image_pct)
    save_histogram(figures / "image_bearing_reviews_per_business.png", dist_rows, n_businesses, n_with_image)
    save_categories(figures / "image_coverage_by_category.png", category_table)

    write_csv(tables / "exploratory_grid_25x25_New_York.csv", grid_tables["25x25"])
    write_csv(tables / "exploratory_grid_9x10_New_York.csv", grid_tables["9x10"])
    write_csv(tables / "category_image_coverage_New_York.csv", category_table)
    write_csv(tables / "image_review_distribution_New_York.csv", dist_rows)
    export = dict(summary)
    export["grids"] = {name: public_grid(item) for name, item in summary["grids"].items()}
    (tables / "feasibility_summary_New_York.json").write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    write_brief(ROOT / "results" / "feasibility_brief_New_York.md", export)
    print(json.dumps({key: export[key] for key in ("n_businesses", "businesses_with_image", "pct_businesses_with_image", "image_bearing_reviews", "unique_image_urls", "coverage_reconciliation")}, indent=2))


if __name__ == "__main__":
    main()
