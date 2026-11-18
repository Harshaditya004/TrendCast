"""Sensitivity: gate learning-rate (is the null caused by under-trained signal logits?) and endurance threshold tau."""
import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A, windows as W
R = A.R; H = W.H
def stack_dict(path):
    d = pickle.load(open(path, 'rb')); folds = sorted(d['res']); keys = list(d['res'][0]['res'].keys())
    out = {k: {q: np.concatenate([d['res'][fi]['res'][k][q] for fi in folds], 1) for q in d['res'][0]['res'][k]} for k in keys}
    meta = {k: np.concatenate([d['res'][fi]['meta'][k] for fi in folds]) for k in d['res'][0]['meta'] if k != 'X'}
    sds = [W.norm_stats(A.P, W.CFG['folds'][fi][0] - H)[2] for fi in folds]; ms = [A.mase_scale(W.CFG['folds'][fi][0] - H) for fi in folds]
    meta['sd_delta'] = np.concatenate([np.full(len(d['res'][fi]['meta']['t']), sds[i]) for i, fi in enumerate(folds)]); meta['mase'] = np.concatenate([np.full(len(d['res'][fi]['meta']['t']), ms[i]) for i, fi in enumerate(folds)])
    return out, meta
if 'gate' in sys.argv[1:]:
    out, meta = stack_dict(f'{R}/gate_lr.pkl'); rows = []
    for k in out:
        df = A.per_seed_metrics(out, meta, k); w = out[k]['w']
        r = dict(model=k[0], setting=k[1], MAE_log=df.MAE_log.mean(), sd=df.MAE_log.std(ddof=1), MAE_end_unc=df.MAE_end_unc.mean(), w_std_across_samples=w.std(1).mean(), w_range_across_signals=(w.mean(1).max(1) - w.mean(1).min(1)).mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(f'{R}/table_gate_lr_sensitivity.csv', index=False); print(pd.DataFrame(rows).round(3).to_string())
    for gm in ['gate_lr_x10', 'gate_lr_x30']:
        c = A.compare(out, meta, ('LPCF (K=3)', gm), ('K=1 (learned fixed w)', gm), 'pop'); print(gm, 'LPCF vs K=1 pop', {k: round(v, 4) if isinstance(v, float) else v for k, v in c.items()})
if 'tau' in sys.argv[1:]:
    rows = []
    for tau, p in [(0.7, f'{R}/rolling_pooled_tau0.7.pkl'), (0.8, f'{R}/rolling_pooled.pkl'), (0.9, f'{R}/rolling_pooled_tau0.9.pkl')]:
        if not os.path.exists(p): continue
        out, meta, _ = A.stack_pooled(p)
        for m in ['LPCF (K=3)', 'LPCF (K=3, no phi->E)', 'K=1 (learned fixed w)', 'Early concat', 'LSTM', 'DLinear', 'Random Forest', 'XGBoost', 'XGBoost-AFT', 'Train-mean']:
            df = A.per_seed_metrics(out, meta, m)
            rows.append(dict(tau=tau, model=m, n_uncens=int((~meta['cen']).sum()), MAE_end_unc=df.MAE_end_unc.mean(), sd=df.MAE_end_unc.std(ddof=1) if len(df) > 1 else 0, Cindex=df.Cindex.mean()))
        c = A.compare(out, meta, 'LPCF (K=3)', 'LPCF (K=3, no phi->E)', 'end'); rows.append(dict(tau=tau, model='[LPCF phi->E gain vs no-phi]', MAE_end_unc=c['delta'], sd=c['lo'], Cindex=c['hi']))
    pd.DataFrame(rows).to_csv(f'{R}/table_tau_sensitivity.csv', index=False); print(pd.DataFrame(rows).round(3).to_string())
