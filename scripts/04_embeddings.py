#!/usr/bin/env python3
"""Review embeddings and county means with all-MiniLM-L6-v2.

The advisors asked for MiniLM review embeddings, a county-level mean of those
embeddings, and, later, cosine similarity with third-place reference sentences.
This script implements only the first two. It does not score third-place text.

Review-level L2 normalization is a methodological choice of this project, not
an explicit instruction from the advisors.

Saved review rows are the mean-pooling outputs, before L2 normalization.
encode(..., normalize_embeddings=False) is not enough for this checkpoint:
all-MiniLM-L6-v2 includes its own Normalize module after pooling, and that
flag does not turn the module off. Encoding skips that module and still passes
normalize_embeddings=False. The county vector is the arithmetic mean of those
saved rows after each row has been L2-normalized. The county mean itself is
not L2-normalized. Sums use float64 and saved vectors use float32.

Cosine similarity, applied later, divides the finished county mean by its
norm. That removes the magnitude of the mean. It does not change the
direction the mean already has, and that direction depends on the magnitudes
of the review vectors that were averaged. A larger raw MiniLM norm pulls the
mean more. Dividing the finished mean by its norm cannot give that influence
back. Normalizing each review first is what gives every embedded review equal
directional weight. Leaving the mean unnormalized also keeps ||mean|| as a
record of how aligned those directions were.

Inputs are read only. Blank text after strip() is skipped. Short text,
duplicate reviews, emoji, and other languages are kept. The only text change
is strip(), so the encoded string is the same string that passed the filter.
County aggregation includes a business only when its crosswalk match_status
is "matched". County names are not hardcoded.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64
DEFAULT_DEVICE = "cpu"
AGGREGATION = "l2_normalize_each_review_then_arithmetic_mean"
INDEX_FIELDS = (
    "row_index",
    "source_line",
    "gmap_id",
    "county_fips",
    "county_name",
    "state_fips",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check(checks, name, passed, detail):
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    return bool(passed)


def input_paths(state):
    processed = ROOT / "data" / "processed"
    return {
        "reviews": processed / f"reviews_{state}.jsonl",
        "businesses": processed / f"businesses_{state}.jsonl",
        "crosswalk": processed / f"county_crosswalk_{state}.jsonl",
    }


def output_paths(state, artifact_dir, summary_dir):
    artifact_dir = Path(artifact_dir)
    summary_dir = Path(summary_dir)
    return {
        "review_embeddings": artifact_dir / f"review_embeddings_{state}.npy",
        "index": artifact_dir / f"review_embedding_index_{state}.jsonl",
        "county": artifact_dir / f"county_embeddings_{state}.npz",
        "summary": summary_dir / f"embedding_summary_{state}.json",
    }


def prepare_review_text(text):
    """Return text.strip(), or None when the review should not be encoded."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if stripped == "":
        return None
    return stripped


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def load_keyed_jsonl(path, label):
    rows = {}
    duplicates = []
    for _, row in iter_jsonl(path):
        gmap_id = row.get("gmap_id")
        if not isinstance(gmap_id, str) or gmap_id == "":
            raise SystemExit(f"{label} has a row without a string gmap_id: {path}")
        if gmap_id in rows:
            duplicates.append(gmap_id)
            continue
        rows[gmap_id] = row
    if duplicates:
        raise SystemExit(f"{label} has duplicate gmap_id values, for example {duplicates[0]}")
    return rows


def review_decision(row, businesses, crosswalk):
    """Decide whether one review joins a matched business and has text.

    Returns ("embed", record) or ("skip", reason).
    """
    text = prepare_review_text(row.get("text"))
    if text is None:
        return "skip", "blank_or_non_string"
    gmap_id = row.get("gmap_id")
    if not isinstance(gmap_id, str) or gmap_id not in businesses:
        return "skip", "missing_business"
    county = crosswalk.get(gmap_id)
    if county is None:
        return "skip", "missing_crosswalk"
    if county.get("match_status") != "matched":
        return "skip", "unmatched"
    county_fips = county.get("county_fips")
    county_name = county.get("county_name")
    state_fips = county.get("state_fips")
    if not isinstance(county_fips, str) or county_fips == "":
        return "skip", "missing_fips"
    if not isinstance(county_name, str) or not isinstance(state_fips, str):
        return "skip", "missing_county_metadata"
    return "embed", {
        "gmap_id": gmap_id,
        "text": text,
        "county_fips": county_fips,
        "county_name": county_name,
        "state_fips": state_fips,
    }


def select_embeddable_reviews(review_path, businesses, crosswalk):
    selected = []
    reasons = Counter()
    for source_line, row in iter_jsonl(review_path):
        kind, payload = review_decision(row, businesses, crosswalk)
        reasons[kind if kind == "embed" else payload] += 1
        if kind != "embed":
            continue
        payload["source_line"] = source_line
        selected.append(payload)
    reasons["reviews_read"] = sum(reasons.values())
    return selected, reasons


def stratified_sample(records, max_reviews, seed):
    """Pick reviews from every county, then restore reviews-file order.

    Within a county, order is SHA-256(seed, county_fips, source_line), with
    source_line as a tie break. Counties are visited in sorted FIPS order,
    one review at a time. This does not depend on random.Random.
    """
    if max_reviews is None:
        return list(records)
    if max_reviews < 1:
        raise SystemExit("--max-reviews must be >= 1")
    if max_reviews >= len(records):
        return list(records)
    by_county = {}
    for index, record in enumerate(records):
        by_county.setdefault(record["county_fips"], []).append(index)
    queues = {}
    for county_fips, indices in by_county.items():
        queues[county_fips] = sorted(
            indices,
            key=lambda index: (
                hashlib.sha256(
                    f"{seed}:{county_fips}:{records[index]['source_line']}".encode()
                ).digest(),
                records[index]["source_line"],
            ),
        )
    cursors = {county_fips: 0 for county_fips in queues}
    picked = []
    remaining = max_reviews
    while remaining > 0:
        progressed = False
        for county_fips in sorted(queues):
            if remaining <= 0:
                break
            cursor = cursors[county_fips]
            if cursor >= len(queues[county_fips]):
                continue
            picked.append(queues[county_fips][cursor])
            cursors[county_fips] = cursor + 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    picked.sort()
    return [records[index] for index in picked]


def county_labels(records):
    labels = {}
    for record in records:
        fips = record["county_fips"]
        label = (record["county_name"], record["state_fips"])
        previous = labels.get(fips)
        if previous is not None and previous != label:
            raise SystemExit(
                f"county {fips} has conflicting metadata: {previous} vs {label}"
            )
        labels[fips] = label
    return labels


def business_counts(businesses, crosswalk, embedded_ids_by_county):
    """Count matched businesses that exist in both the business file and the crosswalk."""
    totals = Counter()
    names = {}
    skipped = Counter()
    for gmap_id in businesses:
        county = crosswalk.get(gmap_id)
        if county is None:
            skipped["business_missing_crosswalk"] += 1
            continue
        if county.get("match_status") != "matched":
            skipped["business_unmatched"] += 1
            continue
        fips = county.get("county_fips")
        if not isinstance(fips, str) or fips == "":
            skipped["business_missing_fips"] += 1
            continue
        totals[fips] += 1
        names[fips] = (county.get("county_name"), county.get("state_fips"))
    with_text = {
        fips: len(ids)
        for fips, ids in embedded_ids_by_county.items()
    }
    return totals, with_text, names, skipped


def l2_normalize_rows(raw):
    """L2-normalize each row. Zero or non-finite rows raise."""
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("expected a non-empty (n, d) embedding matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix has NaN or Inf")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("embedding row has zero or non-finite L2 norm")
    return matrix / norms[:, None], norms


def aggregate_normalized_means(raw, county_fips):
    """Mean of per-row L2-normalized vectors. The mean is not normalized again."""
    fips = [str(value) for value in county_fips]
    if len(fips) != len(raw):
        raise ValueError("county_fips length does not match embedding rows")
    unit, raw_norms = l2_normalize_rows(raw)
    grouped = {}
    for index, county in enumerate(fips):
        grouped.setdefault(county, []).append(index)
    means = {}
    for county in sorted(grouped):
        block = unit[np.asarray(grouped[county], dtype=np.int64)]
        total = np.sum(block, axis=0, dtype=np.float64)
        mean = total / block.shape[0]
        means[county] = {
            "mean": mean,
            "n_reviews": int(block.shape[0]),
            "l2_norm": float(np.linalg.norm(mean)),
            "max_abs_unit_norm_error": float(np.max(np.abs(np.linalg.norm(block, axis=1) - 1.0))),
        }
    return means, raw_norms


def independent_normalized_mean(rows):
    """Same formula as aggregate_normalized_means, written without NumPy."""
    accumulator = None
    count = 0
    for row in rows:
        values = [float(value) for value in row]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("synthetic row is not normalizable")
        unit = [value / norm for value in values]
        if accumulator is None:
            accumulator = unit
        else:
            accumulator = [left + right for left, right in zip(accumulator, unit)]
        count += 1
    return [value / count for value in accumulator]


def vector_close(left, right, tol=1e-12):
    if len(left) != len(right):
        return False
    return all(math.isclose(a, b, rel_tol=0, abs_tol=tol) for a, b in zip(left, right))


def unit_vector(values):
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def run_synthetic_tests(checks):
    """Verify aggregation math and join rules without loading the model."""
    raw = np.array(
        [
            [3.0, 0.0],
            [0.0, 4.0],
            [6.0, 8.0],
            [5.0, 0.0],
        ],
        dtype=np.float64,
    )
    fips = ["00002", "00002", "00001", "00003"]
    means, _ = aggregate_normalized_means(raw, fips)
    expected = {
        "00001": independent_normalized_mean([[6.0, 8.0]]),
        "00002": independent_normalized_mean([[3.0, 0.0], [0.0, 4.0]]),
        "00003": independent_normalized_mean([[5.0, 0.0]]),
    }
    math_ok = True
    details = []
    for county, manual in expected.items():
        produced = means[county]["mean"].tolist()
        match = vector_close(produced, manual)
        math_ok = math_ok and match
        details.append(f"{county} produced={produced} independent={manual}")
    # [3, 0] and [0, 4] have different norms. Their raw mean points at
    # (0.6, 0.8) after its own L2 normalization. The mean of the unit
    # vectors points at (1/sqrt(2), 1/sqrt(2)). Later cosine similarity
    # uses the second normalization only, so it cannot recover the first.
    raw_mean = [1.5, 2.0]
    raw_direction = unit_vector(raw_mean)
    equal_weight_direction = unit_vector(expected["00002"])
    directions_differ = not vector_close(raw_direction, equal_weight_direction, tol=1e-9)
    unequal_mean_norm = abs(means["00002"]["l2_norm"] - math.sqrt(0.5)) < 1e-12
    single_review_norm = abs(means["00003"]["l2_norm"] - 1.0) < 1e-12
    # A tenfold change in magnitude does not move a unit-vector mean.
    scaled = aggregate_normalized_means(
        np.array([[1.0, 0.0], [0.0, 1.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float64),
        ["A", "A", "B", "B"],
    )[0]
    scale_invariant = vector_close(
        scaled["A"]["mean"].tolist(), scaled["B"]["mean"].tolist()
    )
    check(
        checks,
        "synthetic_mean_matches_independent_arithmetic",
        math_ok and unequal_mean_norm and single_review_norm and scale_invariant,
        (
            "; ".join(details)
            + f"; county 00002 l2_norm={means['00002']['l2_norm']}"
            + f"; raw-mean direction={raw_direction}"
            + f"; equal-weight direction={equal_weight_direction}"
        ),
    )
    check(
        checks,
        "synthetic_final_normalization_does_not_replace_review_normalization",
        directions_differ and unequal_mean_norm,
        (
            "mean of raw (3,0) and (0,4) has cosine direction "
            f"{raw_direction}; mean of their unit vectors has cosine direction "
            f"{equal_weight_direction}; saved county l2_norm="
            f"{means['00002']['l2_norm']} rather than 1"
        ),
    )

    businesses = {"keep": {"gmap_id": "keep"}, "blank": {"gmap_id": "blank"}, "out": {"gmap_id": "out"}}
    crosswalk = {
        "keep": {
            "match_status": "matched",
            "county_fips": "11111",
            "county_name": "Keep",
            "state_fips": "11",
        },
        "blank": {
            "match_status": "matched",
            "county_fips": "11111",
            "county_name": "Keep",
            "state_fips": "11",
        },
        "out": {
            "match_status": "unmatched",
            "county_fips": None,
            "county_name": None,
            "state_fips": None,
        },
    }
    samples = [
        {"gmap_id": "keep", "text": "  short  "},
        {"gmap_id": "keep", "text": "  short  "},
        {"gmap_id": "keep", "text": "  "},
        {"gmap_id": "keep", "text": None},
        {"gmap_id": "blank", "text": ""},
        {"gmap_id": "out", "text": "should skip"},
        {"gmap_id": "missing", "text": "should skip"},
        {"gmap_id": "keep", "text": "🙂 kept"},
    ]
    decisions = [review_decision(row, businesses, crosswalk) for row in samples]
    embedded = [payload for kind, payload in decisions if kind == "embed"]
    kept_text = [item["text"] for item in embedded]
    selection_ok = kept_text == ["short", "short", "🙂 kept"] and all(
        item["county_fips"] == "11111" for item in embedded
    )
    check(
        checks,
        "synthetic_selection_keeps_short_duplicate_and_emoji_text",
        selection_ok,
        f"embedded texts={kept_text}",
    )

    population = []
    for county_fips, count in (("00002", 5), ("00001", 2), ("00003", 4)):
        for offset in range(count):
            population.append(
                {
                    "county_fips": county_fips,
                    "source_line": len(population) + 1,
                    "marker": f"{county_fips}-{offset}",
                }
            )
    sampled = stratified_sample(population, max_reviews=6, seed=42)
    sampled_counties = {record["county_fips"] for record in sampled}
    file_order = [record["source_line"] for record in sampled]
    check(
        checks,
        "synthetic_sample_covers_every_county_and_keeps_file_order",
        sampled_counties == {"00001", "00002", "00003"}
        and len(sampled) == 6
        and file_order == sorted(file_order),
        f"counties={sorted(sampled_counties)} source_lines={file_order}",
    )


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")
        return "cuda"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested, but it is not available in this torch build.")
        return "mps"
    raise SystemExit(f"Unsupported device {requested!r}. Use cpu, cuda, or mps.")


def load_model(device):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is not installed. "
            "Install the versions listed in requirements.txt. "
            "This script does not substitute another embedding model."
        ) from exc
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load {MODEL_NAME} on device {device}. "
            "No fallback model was loaded. "
            f"Underlying error: {exc}"
        ) from exc
    if hasattr(model, "get_embedding_dimension"):
        dimension = int(model.get_embedding_dimension())
    else:
        dimension = int(model.get_sentence_embedding_dimension())
    if dimension != EMBEDDING_DIM:
        raise SystemExit(
            f"{MODEL_NAME} reported dimension {dimension}, expected {EMBEDDING_DIM}. "
            "No fallback model was loaded."
        )
    return model


def model_revision(model):
    try:
        module = model[0]
        auto_model = getattr(module, "auto_model", None)
        revision = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
        if revision:
            return revision, "transformers.config._commit_hash"
    except Exception:
        pass
    try:
        from huggingface_hub import model_info

        info = model_info(MODEL_NAME)
        if info.sha:
            return info.sha, "huggingface_hub.model_info"
    except Exception as exc:
        return None, f"unavailable: {exc}"
    return None, "unavailable"


def module_names(model):
    return [type(module).__name__ for module in model]


def _skip_builtin_normalize(model):
    """Disable checkpoint Normalize modules for one encode call.

    Returns the list of original forward methods so the caller can restore them.
    """
    replaced = []
    for module in model:
        if type(module).__name__ != "Normalize":
            continue

        def passthrough(features, *args, **kwargs):
            return features

        replaced.append((module, module.forward))
        module.forward = passthrough
    return replaced


def encode_reviews(model, texts, batch_size, device):
    replaced = _skip_builtin_normalize(model)
    if not replaced:
        raise SystemExit(
            f"{MODEL_NAME} loaded without a Normalize module ({module_names(model)}). "
            "The raw-output path for this checkpoint skips that module. "
            "No fallback model was used."
        )
    try:
        try:
            encoded = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
                device=device,
            )
        except Exception as exc:
            raise SystemExit(
                f"Inference with {MODEL_NAME} failed. No fallback model was used. Underlying error: {exc}"
            ) from exc
    finally:
        for module, original in replaced:
            module.forward = original
    matrix = np.asarray(encoded, dtype=np.float32)
    if matrix.shape != (len(texts), EMBEDDING_DIM):
        raise SystemExit(
            f"Encoder returned shape {matrix.shape}, expected {(len(texts), EMBEDDING_DIM)}."
        )
    return matrix


def count_truncated(model, texts):
    """Count reviews whose token ids exceed the model max length.

    The tokenizer's own max length is raised only for this count. Encoding
    itself still truncates at model.max_seq_length.
    """
    tokenizer = model.tokenizer
    limit = int(model.max_seq_length)
    previous_limit = tokenizer.model_max_length
    tokenizer.model_max_length = max(int(previous_limit), 1_000_000)
    truncated = 0
    step = 256
    try:
        for start in range(0, len(texts), step):
            batch = texts[start : start + step]
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            truncated += sum(len(token_ids) > limit for token_ids in encoded["input_ids"])
    finally:
        tokenizer.model_max_length = previous_limit
    return truncated


def write_index(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_index, record in enumerate(records):
            payload = {
                "row_index": row_index,
                "source_line": record["source_line"],
                "gmap_id": record["gmap_id"],
                "county_fips": record["county_fips"],
                "county_name": record["county_name"],
                "state_fips": record["state_fips"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_county_npz(path, county_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.stack([row["mean"] for row in county_rows]).astype(np.float32, copy=False)
    np.savez(
        path,
        county_fips=np.array([row["county_fips"] for row in county_rows], dtype="U16"),
        county_name=np.array([row["county_name"] for row in county_rows], dtype="U64"),
        state_fips=np.array([row["state_fips"] for row in county_rows], dtype="U8"),
        embeddings=embeddings,
        n_businesses_total=np.array([row["n_businesses_total"] for row in county_rows], dtype=np.int64),
        n_businesses_with_text=np.array(
            [row["n_businesses_with_text"] for row in county_rows], dtype=np.int64
        ),
        n_reviews_embedded=np.array([row["n_reviews_embedded"] for row in county_rows], dtype=np.int64),
        l2_norm=np.array([row["l2_norm"] for row in county_rows], dtype=np.float64),
        aggregation=np.array(AGGREGATION),
    )
    return embeddings


def load_index(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_county_rows(records, means, totals, with_text, names):
    labels = county_labels(records)
    for fips, label in names.items():
        previous = labels.get(fips)
        if previous is not None and previous != label:
            raise SystemExit(f"county {fips} metadata disagrees between reviews and businesses")
        labels.setdefault(fips, label)
    embedded_counts = Counter(record["county_fips"] for record in records)
    rows = []
    for fips in sorted(labels):
        name, state_fips = labels[fips]
        mean_info = means.get(fips)
        rows.append(
            {
                "county_fips": fips,
                "county_name": name,
                "state_fips": state_fips,
                "n_businesses_total": int(totals.get(fips, 0)),
                "n_businesses_with_text": int(with_text.get(fips, 0)),
                "n_reviews_embedded": int(embedded_counts.get(fips, 0)),
                "l2_norm": None if mean_info is None else mean_info["l2_norm"],
                "mean": None if mean_info is None else mean_info["mean"],
                "max_abs_unit_norm_error": None
                if mean_info is None
                else mean_info["max_abs_unit_norm_error"],
            }
        )
    return rows


def validate_outputs(
    checks,
    paths,
    records,
    businesses,
    crosswalk,
    review_path,
    saved_raw,
    model,
    batch_size,
    device,
    population_business_totals,
):
    index_rows = load_index(paths["index"])
    loaded_raw = np.load(paths["review_embeddings"])
    with np.load(paths["county"]) as archive:
        saved_fips = [str(value) for value in archive["county_fips"].tolist()]
        saved_embeddings = archive["embeddings"]
        saved_review_counts = archive["n_reviews_embedded"]
        saved_business_totals = archive["n_businesses_total"]
        saved_business_with_text = archive["n_businesses_with_text"]
        saved_norms = archive["l2_norm"]
        saved_aggregation = str(archive["aggregation"])

    check(
        checks,
        "saved_raw_shape_and_dtype",
        loaded_raw.shape == saved_raw.shape == (len(records), EMBEDDING_DIM)
        and loaded_raw.dtype == np.float32
        and saved_raw.dtype == np.float32,
        f"shape={tuple(loaded_raw.shape)} dtype={loaded_raw.dtype}",
    )
    finite = bool(np.isfinite(loaded_raw).all() and np.isfinite(saved_embeddings).all())
    check(checks, "embeddings_finite", finite, f"raw_and_county_finite={finite}")
    check(
        checks,
        "index_rows_unique_and_aligned",
        len(index_rows) == len(records) == loaded_raw.shape[0]
        and [row["row_index"] for row in index_rows] == list(range(len(records)))
        and [row["gmap_id"] for row in index_rows] == [record["gmap_id"] for record in records]
        and [row["county_fips"] for row in index_rows] == [record["county_fips"] for record in records]
        and [row["source_line"] for row in index_rows] == [record["source_line"] for record in records],
        f"index_rows={len(index_rows)} embedding_rows={loaded_raw.shape[0]}",
    )

    needed_lines = {row["source_line"] for row in index_rows}
    found = {}
    with Path(review_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number in needed_lines:
                found[line_number] = json.loads(line)
    alignment_failures = []
    for row in index_rows:
        source = found.get(row["source_line"])
        county = crosswalk.get(row["gmap_id"])
        business_ok = row["gmap_id"] in businesses
        text = None if source is None else prepare_review_text(source.get("text"))
        if (
            source is None
            or source.get("gmap_id") != row["gmap_id"]
            or text is None
            or not business_ok
            or county is None
            or county.get("match_status") != "matched"
            or county.get("county_fips") != row["county_fips"]
            or county.get("county_name") != row["county_name"]
            or county.get("state_fips") != row["state_fips"]
        ):
            alignment_failures.append(row["row_index"])
    check(
        checks,
        "review_to_county_alignment",
        not alignment_failures,
        f"misaligned_rows={alignment_failures[:10]} checked={len(index_rows)}",
    )

    recomputed, raw_norms = aggregate_normalized_means(
        loaded_raw, [row["county_fips"] for row in index_rows]
    )
    expected_fips = sorted({record["county_fips"] for record in records})
    recompute_error = 0.0
    count_ok = saved_fips == expected_fips
    for index, fips in enumerate(saved_fips):
        if fips not in recomputed:
            count_ok = False
            continue
        produced = recomputed[fips]["mean"].astype(np.float32)
        recompute_error = max(recompute_error, float(np.max(np.abs(produced - saved_embeddings[index]))))
        if int(saved_review_counts[index]) != recomputed[fips]["n_reviews"]:
            count_ok = False
        if abs(float(saved_norms[index]) - recomputed[fips]["l2_norm"]) > 1e-8:
            count_ok = False
    unit_error = max(info["max_abs_unit_norm_error"] for info in recomputed.values())
    raw_norm_gap = float(np.max(np.abs(raw_norms - 1.0)))
    check(
        checks,
        "saved_raw_embeddings_keep_magnitude",
        raw_norm_gap > 1e-3,
        (
            f"max_abs_norm_minus_one={raw_norm_gap} "
            f"raw_l2_min={float(np.min(raw_norms))} raw_l2_max={float(np.max(raw_norms))}"
        ),
    )
    check(
        checks,
        "per_review_l2_and_county_mean",
        count_ok
        and recompute_error == 0.0
        and unit_error < 1e-8
        and saved_embeddings.shape == (len(expected_fips), EMBEDDING_DIM)
        and saved_embeddings.dtype == np.float32
        and saved_aggregation == AGGREGATION,
        (
            f"max_abs_county_diff={recompute_error} max_abs_unit_norm_error={unit_error} "
            f"county_shape={tuple(saved_embeddings.shape)} dtype={saved_embeddings.dtype}"
        ),
    )

    by_county_ids = {}
    for record in records:
        by_county_ids.setdefault(record["county_fips"], set()).add(record["gmap_id"])
    count_failures = []
    for index, fips in enumerate(saved_fips):
        expected_total = int(population_business_totals.get(fips, 0))
        expected_with_text = len(by_county_ids.get(fips, ()))
        expected_reviews = sum(record["county_fips"] == fips for record in records)
        if (
            int(saved_business_totals[index]) != expected_total
            or int(saved_business_with_text[index]) != expected_with_text
            or int(saved_review_counts[index]) != expected_reviews
        ):
            count_failures.append(fips)
    check(
        checks,
        "county_review_and_business_counts",
        not count_failures and sum(int(value) for value in saved_review_counts) == len(records),
        f"mismatching_counties={count_failures}",
    )

    reencode_positions = rows_to_reencode(len(records))
    reencode_texts = []
    for position in reencode_positions:
        source_line = index_rows[position]["source_line"]
        reencode_texts.append(prepare_review_text(found[source_line]["text"]))
    recomputed_rows = encode_reviews(model, reencode_texts, batch_size, device)
    original_rows = loaded_raw[reencode_positions]
    max_abs = float(np.max(np.abs(recomputed_rows.astype(np.float64) - original_rows.astype(np.float64))))
    check(
        checks,
        "reencoded_rows_match_saved_raw_embeddings",
        max_abs <= 1e-5,
        f"rows_reencoded={len(reencode_positions)} max_abs_diff={max_abs}",
    )
    return {
        "raw_l2_norm_min": float(np.min(raw_norms)),
        "raw_l2_norm_max": float(np.max(raw_norms)),
        "raw_l2_norm_mean": float(np.mean(raw_norms)),
        "max_abs_county_recompute_diff": recompute_error,
        "max_abs_unit_norm_error": unit_error,
        "reencode_rows": len(reencode_positions),
        "reencode_max_abs_diff": max_abs,
    }


def rows_to_reencode(n_rows):
    if n_rows <= 400:
        return list(range(n_rows))
    picks = {0, n_rows - 1}
    step = max(1, n_rows // 16)
    picks.update(range(0, n_rows, step))
    return sorted(picks)


def methodology_block():
    return {
        "review_embeddings_saved": (
            "mean-pooling outputs before L2 normalization. The checkpoint Normalize "
            "module is skipped, and encode is called with normalize_embeddings=False. "
            "That flag alone does not disable the checkpoint module."
        ),
        "county_aggregation": AGGREGATION,
        "county_mean_l2_normalized_before_save": False,
        "aggregation_dtype": "float64",
        "saved_dtype": "float32",
        "review_level_normalization_authority": (
            "Methodological choice for this project, not an explicit instruction from the advisors. "
            "The advisors requested MiniLM review embeddings, county-level means, and later cosine "
            "similarity with third-place reference sentences."
        ),
        "why_cosine_normalization_does_not_make_review_normalization_redundant": (
            "Cosine similarity divides each compared vector by its own norm. On a county mean, "
            "that cancels the magnitude of the finished mean and leaves its direction. The direction "
            "is fixed by how the mean was computed. Averaging raw embeddings weights each review by "
            "its MiniLM vector norm. Averaging L2-normalized review embeddings gives each review equal "
            "directional weight. Those two means generally point different ways, so cosine similarity "
            "against the same reference sentence scores them differently. Normalizing the mean after "
            "the fact cannot reconstruct the equal-weight direction."
        ),
    }


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default="New_York")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    checks = []
    run_synthetic_tests(checks)
    if args.self_test:
        passed = all(item["passed"] for item in checks)
        for item in checks:
            print(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}: {item['detail']}")
        if not passed:
            raise SystemExit(1)
        print("Synthetic checks passed. No model was loaded.")
        return

    default_artifact = (ROOT / "data" / "processed").resolve()
    default_summary = (ROOT / "results" / "tables").resolve()
    if args.max_reviews is not None and (
        args.artifact_dir.resolve() == default_artifact or args.summary_dir.resolve() == default_summary
    ):
        raise SystemExit(
            "A sampled run uses the same filenames as a full run. "
            "Pass both --artifact-dir and --summary-dir so smoke outputs stay separate."
        )

    paths_in = input_paths(args.state)
    for path in paths_in.values():
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")
    hashes_before = {name: sha256_file(path) for name, path in paths_in.items()}

    businesses = load_keyed_jsonl(paths_in["businesses"], "businesses")
    crosswalk = load_keyed_jsonl(paths_in["crosswalk"], "crosswalk")
    selected, reasons = select_embeddable_reviews(paths_in["reviews"], businesses, crosswalk)
    if not selected:
        raise SystemExit("No embeddable reviews. The model was not loaded.")
    records = stratified_sample(selected, args.max_reviews, args.sample_seed)
    if args.max_reviews is not None and len({record["county_fips"] for record in records}) < 2:
        raise SystemExit("The sample contains fewer than two counties. No embeddings were written.")

    device = resolve_device(args.device)
    model = load_model(device)
    revision, revision_source = model_revision(model)
    texts = [record["text"] for record in records]
    raw = encode_reviews(model, texts, args.batch_size, device)
    truncated = count_truncated(model, texts)
    means, _ = aggregate_normalized_means(raw, [record["county_fips"] for record in records])

    embedded_ids = {}
    for record in records:
        embedded_ids.setdefault(record["county_fips"], set()).add(record["gmap_id"])
    totals, with_text, names, business_skip = business_counts(businesses, crosswalk, embedded_ids)
    population_reviews = Counter(record["county_fips"] for record in selected)
    county_rows = build_county_rows(records, means, totals, with_text, names)
    for row in county_rows:
        row["n_reviews_embeddable_population"] = int(population_reviews.get(row["county_fips"], 0))
    represented = [row for row in county_rows if row["mean"] is not None]

    paths_out = output_paths(args.state, args.artifact_dir, args.summary_dir)
    paths_out["review_embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths_out["review_embeddings"], raw)
    write_index(paths_out["index"], records)
    write_county_npz(paths_out["county"], represented)

    hashes_after = {name: sha256_file(path) for name, path in paths_in.items()}
    unchanged = hashes_before == hashes_after
    check(
        checks,
        "input_files_unchanged",
        unchanged,
        " ".join(f"{name}={hashes_before[name]}" for name in hashes_before),
    )
    diagnostics = validate_outputs(
        checks,
        paths_out,
        records,
        businesses,
        crosswalk,
        paths_in["reviews"],
        raw,
        model,
        args.batch_size,
        device,
        totals,
    )
    passed = all(item["passed"] for item in checks)
    summary = {
        "state": args.state,
        "run_type": "full" if args.max_reviews is None else "smoke",
        "model": {
            "name": MODEL_NAME,
            "revision": revision,
            "revision_source": revision_source,
            "modules": module_names(model),
            "raw_output": "skip checkpoint Normalize module, then encode with normalize_embeddings=False",
            "embedding_dimension": EMBEDDING_DIM,
            "max_seq_length": int(model.max_seq_length),
            "normalize_embeddings": False,
            "n_reviews_truncated_to_max_seq_length": truncated,
        },
        "dependencies": {
            "sentence_transformers": package_version("sentence-transformers"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "numpy": package_version("numpy"),
            "huggingface_hub": package_version("huggingface-hub"),
        },
        "configuration": {
            "batch_size": args.batch_size,
            "device": device,
            "max_reviews": args.max_reviews,
            "sample_seed": args.sample_seed,
            "sampling": (
                "all embeddable matched reviews in reviews-file order"
                if args.max_reviews is None
                else (
                    "round-robin over sorted county FIPS; within a county, "
                    "SHA-256(sample_seed, county_fips, source_line); selected rows "
                    "rewritten in reviews-file order"
                )
            ),
        },
        "methodology": methodology_block(),
        "inputs": {
            name: {"path": rel(path), "sha256": hashes_before[name]}
            for name, path in paths_in.items()
        },
        "counts": {
            "reviews_read": reasons["reviews_read"],
            "reviews_embeddable_matched": reasons["embed"],
            "reviews_blank_or_non_string": reasons["blank_or_non_string"],
            "reviews_missing_business": reasons["missing_business"],
            "reviews_missing_crosswalk": reasons["missing_crosswalk"],
            "reviews_unmatched_business": reasons["unmatched"],
            "reviews_missing_fips": reasons["missing_fips"],
            "reviews_embedded_this_run": len(records),
            "counties_in_county_matrix": len(represented),
            "business_skip": dict(business_skip),
            **diagnostics,
        },
        "counties": [
            {
                "county_fips": row["county_fips"],
                "county_name": row["county_name"],
                "state_fips": row["state_fips"],
                "n_businesses_total": row["n_businesses_total"],
                "n_businesses_with_text": row["n_businesses_with_text"],
                "n_reviews_embedded": row["n_reviews_embedded"],
                "n_reviews_embeddable_population": row["n_reviews_embeddable_population"],
                "l2_norm": row["l2_norm"],
                "included_in_county_matrix": row["mean"] is not None,
            }
            for row in county_rows
        ],
        "validation": {"passed": passed, "checks": checks},
        "outputs": {
            "review_embeddings": {
                "path": rel(paths_out["review_embeddings"]),
                "shape": list(raw.shape),
                "dtype": "float32",
                "bytes": paths_out["review_embeddings"].stat().st_size,
            },
            "review_embedding_index": {
                "path": rel(paths_out["index"]),
                "rows": len(records),
                "bytes": paths_out["index"].stat().st_size,
            },
            "county_embeddings": {
                "path": rel(paths_out["county"]),
                "shape": [len(represented), EMBEDDING_DIM],
                "dtype": "float32",
                "bytes": paths_out["county"].stat().st_size,
            },
            "summary": {"path": rel(paths_out["summary"])},
        },
    }
    write_summary(paths_out["summary"], summary)
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}: {item['detail']}")
    print(
        json.dumps(
            {
                "run_type": summary["run_type"],
                "reviews_embedded_this_run": len(records),
                "counties_in_county_matrix": len(represented),
                "validation_passed": passed,
                "model_revision": revision,
            },
            indent=2,
        )
    )
    print(f"Wrote {paths_out['review_embeddings']}")
    print(f"Wrote {paths_out['index']}")
    print(f"Wrote {paths_out['county']}")
    print(f"Wrote {paths_out['summary']}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
