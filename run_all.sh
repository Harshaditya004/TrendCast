#!/bin/sh
# Rebuilds every number in the Review-03 report from the raw files in data/raw (about 25 min on 2 CPU cores).
set -e
cd "$(dirname "$0")/src"
python3 -I run_rolling.py pooled ../results/rolling_pooled.pkl 0.8        # 8 seeds, 3 rolling-origin folds, 19 models
python3 -I run_rolling.py logo ../results/rolling_logo.pkl 0.8            # leave-one-game-out, 5 seeds
python3 -I run_rolling.py signal_ablation ../results/signal_ablation.pkl 0.8
python3 -I run_rolling.py gate_lr ../results/gate_lr.pkl 0.8
python3 -I run_rolling.py pooled ../results/rolling_pooled_tau0.7.pkl 0.7
python3 -I run_rolling.py pooled ../results/rolling_pooled_tau0.9.pkl 0.9
python3 -I run_online.py ../results/online.pkl                            # prequential, equal-update
python3 -I analysis.py && python3 -I analysis2.py && python3 -I analysis3.py gate tau && python3 -I figures.py
