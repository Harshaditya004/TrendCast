# TrendCast / CPAF-Net with LPCF — Review-03 reproducibility package

Everything here is computed from the raw Steam files in `data/raw` (checksums in `data/CHECKSUMS.sha256`). `./run_all.sh` rebuilds all results.

## Layout
- `src/data.py` signals + endurance target (running peak is past-only; right-censoring flag)
- `src/windows.py` windows, training-only normalisation, cut-off-aware labels, rolling-origin folds
- `src/models.py` LPCF (K phases), K=1, uniform, early-concat, TFT-style selection, LSTM, DLinear, TFT-lite; common heads/loss
- `src/baselines.py` Persistence, Drift, Linear, Ridge, RandomForest, XGBoost, XGBoost-AFT
- `src/run_rolling.py`, `src/run_online.py` experiments; `src/analysis*.py`, `src/figures.py` tables/figures
- `results/` all tables (CSV), figures, raw predictions (pkl); `logs/` run logs; `Review03_Results.docx` the write-up

## Known gaps (cannot be closed from the data supplied)
1. **Google Trends CSVs are not in the archive** (and were not reachable from the analysis environment), so only 4 signals exist: popularity, engagement, rating, review growth. `search_interest` is absent, not imputed. To add it: export each `trends_raw.csv` as in `trends_manual_instructions.txt`, add a 5th column to `windows.Panel.F` / `SIG4`, and change `S=4` / `cols=[0,1,2,3]` in `run_*.py` (not implemented here).
2. Only 5 games x 89 days (2026-07-09..2026-10-05). More games/longer histories require new Steam downloads.
3. `trails_sc` has 57 review days of 89; absent days are treated as zero-review days (assumption, not verified).
## Protocol summary
Rolling origin, forecast origin t, window L=14, horizon 7. Folds: test targets in days [55,65), [65,75), [75,89). Training windows need popularity targets known by the fold start; endurance labels for training are re-derived from the series truncated at that date (censored when unresolved). Normalisation uses the known history only. Early stopping uses the last 7 origin days before the fold. Seeds 0-7 (pooled) / 0-4 (others). Paired 7-day moving-block bootstrap (2000 reps) over origin days.
