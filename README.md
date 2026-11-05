# Neighborhood similarity search

Given a place you know (e.g. Rice Village in Houston), find the most similar areas in another city.
Output is a shortlist of areas to visit. See `CLAUDE.md` for goals, design decisions, data sources, and build order.

## Layout
- `notebooks/` - notebooks meant for Kaggle/Colab (data coverage, prep, pretraining, query)
- `src/` - small shared helpers (grid, config)
- `data/` - local scratch only; big data is downloaded on the remote machine
- `outputs/` - local scratch; real artifacts go to Kaggle dataset / HF Hub / Drive

## Rules
- No heavy compute locally. Download big datasets with `wget` on the remote machine.
- Cache each expensive step in chunks and persist outputs off-laptop.

## Local setup (light work only)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
