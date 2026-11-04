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
python scripts/03_preprocess.py --state New_York --sample-size 10000 --max-reviews-per-business 20 --seed 42
python scripts/07_validate_images.py --state New_York --sample-size 200 --seed 42
```

The full New York review file is large; download only after checking disk space. The downloader supports `--kind metadata`, `reviews`, or `both`, streaming downloads and resume via HTTP Range where supported. If the server rejects a request, follow its error output rather than assuming the download completed.

`02_explore.py` creates `results/tables/metadata_summary_New_York.json`. `03_preprocess.py` draws a seeded uniform sample of businesses inside an approximate NYC bounding box (**not** census tract boundaries), then keeps at most 20 reviews per business with a seeded reservoir sample. It streams the gzipped review file and writes JSONL under `data/processed/` plus `results/tables/coverage_New_York.json`. `--max-businesses` still overrides `--sample-size` if you pass it.

`07_validate_images.py` samples image URLs from the processed reviews, downloads them with timeouts and retries, and checks that the bytes decode as images. The CSV report is `results/tables/image_validation_New_York.csv`. Saved images go to `data/processed/sample_images/`.

`08_feasibility_coverage.py` reads the processed JSONL files already on disk. It writes aggregate tables, figures, and `results/feasibility_brief_New_York.md`. It does not download data, rerun sampling, or fit embeddings. The repository has no census-tract boundaries, so the geographic summary uses an exploratory grid over the sampling box. Those cells are not neighborhoods.

Do not commit raw data, processed reviews, sample images, or personal identifiers to GitHub. Those paths are listed in `.gitignore`.

## Next research stages

`04_embeddings.py`, `05_clustering.py`, `06_evaluate.py` are deliberately TODO placeholders. First validate image URL accessibility and decide the geographic unit, then implement these stages.

## Attribution

If using this dataset, cite the two papers specified by its authors on the source page (UCTopic, ACL 2022; Personalized Showcases, SIGIR 2023). Check dataset and image usage rights before redistribution.
