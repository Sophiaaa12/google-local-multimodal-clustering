# Google Local Multimodal Neighborhood Research

Reproducible feasibility study for UCSD Google Local Data (2021). Raw data is not committed.

Source: https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_download.py --state New_York --kind metadata
python scripts/02_explore.py --state New_York
python scripts/01_download.py --state New_York --kind reviews
python scripts/03_preprocess.py --state New_York --max-businesses 5000 --max-reviews-per-business 20
```

The full New York review file is large; download only after checking disk space. The downloader supports `--kind metadata`, `reviews`, or `both`, streaming downloads and resume via HTTP Range where supported. If the server rejects a request, follow its error output rather than assuming the download completed.

`02_explore.py` creates `results/tables/metadata_summary_New_York.json`. `03_preprocess.py` creates sampled JSONL files under `data/processed/` and a coverage summary. Geographic selection is an approximate NYC bounding box, **not** census tract boundaries. Sampling is deterministic by sorted `gmap_id`; it is not statistically representative. Do not commit raw data, review text, image URLs, or personal identifiers to GitHub.

## Next research stages

`04_embeddings.py`, `05_clustering.py`, `06_evaluate.py` are deliberately TODO placeholders. First validate image URL accessibility and decide the geographic unit, then implement these stages.

## Attribution

If using this dataset, cite the two papers specified by its authors on the source page (UCTopic, ACL 2022; Personalized Showcases, SIGIR 2023). Check dataset and image usage rights before redistribution.
