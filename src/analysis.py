"""Aggregate all result pickles into tables (CSV/JSON).  usage: python analysis.py"""
import sys, os, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import windows as W
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'); os.makedirs(R, exist_ok=True)
P = W.Panel(tau=0.8); H = W.H; KC = W.KCAP
rng = np.random.default_rng(12345)
NB = 2000; BLOCK = 7


# ---------------------------------------------------------------- metrics
def cindex(pred, obs, cen):
    """Harrell C for right-censored endurance: pair (i,j) comparable if obs_i<obs_j and i uncensored."""
    num = den = 0.0
    for i in range(len(obs)):
        if cen[i]: continue
        m = obs > obs[i]
        den += m.sum(); num += (pred[m] > pred[i]).sum() + 0.5 * (pred[m] == pred[i]).sum()
    return num / den if den else np.nan


def pop_arrays(meta, dpop, sd_delta):
    """scalar log-scale prediction and count-scale prediction; dpop shape [N] in normalised units"""
    phat = meta['pop_last'] + dpop * sd_delta
    return phat, np.expm1(phat)


def pop_metrics(meta, dpop, sd_delta, mase_scale):
    phat, chat = pop_arrays(meta, dpop, sd_delta); e = phat - meta['pop_true']
    d = meta['pop_true'] - meta['pop_last']; sse = (e ** 2).sum(); sst = ((d - d.mean()) ** 2).sum()
    return dict(MAE_log=np.abs(e).mean(), RMSE_log=np.sqrt((e ** 2).mean()), MASE=np.abs(e).mean() / mase_scale,
                WAPE_cnt=np.abs(chat - meta['cnt_target']).sum() / meta['cnt_target'].sum(),
                sMAPE_cnt=100 * np.mean(2 * np.abs(chat - meta['cnt_target']) / (np.abs(chat) + np.abs(meta['cnt_target']))),
                R2_delta=1 - sse / sst)


def end_metrics(meta, pend):
    unc = ~meta['cen']; e = pend - meta['end']
    viol = ((pend < meta['end']) & meta['cen']).sum() / max(meta['cen'].sum(), 1)    # predicted below a KNOWN lower bound
    return dict(MAE_end_unc=np.abs(e[unc]).mean() if unc.any() else np.nan, RMSE_end_unc=np.sqrt((e[unc] ** 2).mean()) if unc.any() else np.nan,
                Cindex=cindex(pend, meta['end'], meta['cen']), LB_violation=viol)


def mase_scale(t0):
    d = []
    for g in range(len(P.games)):
        pop = P.F[g][:, 0]; d.append(np.abs(pop[6 + H:t0 + 1] - pop[6:t0 + 1 - H]))
    return np.concatenate(d).mean()


# ---------------------------------------------------------------- stack folds
def stack_pooled(path):
    d = pickle.load(open(path, 'rb')); folds = sorted(d['res']); out = {}; meta = {}
    sds, ms, ts = [], [], []
    for fi in folds:
        t0 = W.CFG['folds'][fi][0] - H; sds.append(W.norm_stats(P, t0)[2]); ms.append(mase_scale(t0))
        m = d['res'][fi]['meta']; ts.append(np.full(len(m['t']), fi))
    meta = {k: np.concatenate([d['res'][fi]['meta'][k] for fi in folds]) for k in d['res'][folds[0]]['meta'] if k != 'X'}
    meta['fold'] = np.concatenate(ts); meta['sd_delta'] = np.concatenate([np.full(len(d['res'][fi]['meta']['t']), sds[i]) for i, fi in enumerate(folds)])
    meta['mase'] = np.concatenate([np.full(len(d['res'][fi]['meta']['t']), ms[i]) for i, fi in enumerate(folds)])
    meta['X'] = np.concatenate([d['res'][fi]['meta']['X'] for fi in folds])
    for name in d['res'][folds[0]]['res']:
        out[name] = {k: np.concatenate([d['res'][fi]['res'][name][k] for fi in folds], axis=1) for k in d['res'][folds[0]]['res'][name]}
    return out, meta, d['seeds']


def abs_err_pop(meta, dp):          # [seeds,N]
    return np.abs(meta['pop_last'][None] + dp * meta['sd_delta'][None] - meta['pop_true'][None])


def per_seed_metrics(out, meta, name):
    rows = []
    n_s = out[name]['dpop'].shape[0]
    for s in range(n_s):
        r = {}
        # pooled across folds: normalise MAE with per-sample fold scale
        phat = meta['pop_last'] + out[name]['dpop'][s] * meta['sd_delta']; e = phat - meta['pop_true']
        chat = np.expm1(phat); d = meta['pop_true'] - meta['pop_last']
        r['MAE_log'] = np.abs(e).mean(); r['RMSE_log'] = np.sqrt((e ** 2).mean()); r['MASE'] = (np.abs(e) / meta['mase']).mean()
        r['WAPE_cnt'] = np.abs(chat - meta['cnt_target']).sum() / meta['cnt_target'].sum()
        r['sMAPE_cnt'] = 100 * np.mean(2 * np.abs(chat - meta['cnt_target']) / (np.abs(chat) + np.abs(meta['cnt_target'])))
        r['R2_delta'] = 1 - (e ** 2).sum() / ((d - d.mean()) ** 2).sum()
        r.update(end_metrics(meta, out[name]['end'][s])); rows.append(r)
    return pd.DataFrame(rows)


def block_boot(diff_by_sample, days, blocks=BLOCK, nb=NB, mask=None):
    """Moving-block bootstrap over origin days (blocks of 7 days, all games resampled together). Returns mean diff and 95% CI."""
    x = diff_by_sample if mask is None else diff_by_sample[mask]; dd = days if mask is None else days[mask]
    ud = np.unique(dd); idx = {u: np.where(dd == u)[0] for u in ud}; nd = len(ud)
    if nd < 2 or len(x) == 0: return np.nan, np.nan, np.nan
    starts = np.arange(nd - blocks + 1) if nd >= blocks else np.array([0]); res = []
    nblk = int(np.ceil(nd / min(blocks, nd)))
    for _ in range(nb):
        pick = rng.choice(starts, nblk); sel = np.concatenate([idx[u] for st in pick for u in ud[st:st + blocks]])
        res.append(x[sel].mean())
    return x.mean(), np.percentile(res, 2.5), np.percentile(res, 97.5)


def compare(out, meta, a, b, target='pop'):
    """mean(err_b) - mean(err_a): positive => a better.  Seed-averaged paired errors."""
    if target == 'pop':
        ea, eb = abs_err_pop(meta, out[a]['dpop']).mean(0), abs_err_pop(meta, out[b]['dpop']).mean(0); mask = None
    else:
        ea, eb = np.abs(out[a]['end'] - meta['end']).mean(0), np.abs(out[b]['end'] - meta['end']).mean(0); mask = ~meta['cen']
    m, lo, hi = block_boot(eb - ea, meta['t'], mask=mask)
    # seed-level win count
    if target == 'pop':
        sa = abs_err_pop(meta, out[a]['dpop']).mean(1); sb = abs_err_pop(meta, out[b]['dpop']).mean(1)
    else:
        u = ~meta['cen']; sa = np.abs(out[a]['end'] - meta['end'])[:, u].mean(1); sb = np.abs(out[b]['end'] - meta['end'])[:, u].mean(1)
    wins = int((sa[:, None] < sb[None, :]).sum()) if len(sa) > 1 or len(sb) > 1 else int(sa[0] < sb[0]); tot = len(sa) * len(sb)
    return dict(delta=m, lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0), pairwise_win_frac=wins / tot)


def fmt(m, s): return f'{m:.3f} ± {s:.3f}'


if __name__ == '__main__':
    summary = {}
    # ============================ 1. pooled chronological rolling-origin
    out, meta, seeds = stack_pooled(f'{R}/rolling_pooled.pkl')
    summary['n_test'] = int(len(meta['t'])); summary['n_test_end_uncens'] = int((~meta['cen']).sum()); summary['seeds'] = seeds
    rows = []
    for name in out:
        df = per_seed_metrics(out, meta, name); r = dict(model=name, n_seeds=len(df))
        for c in df.columns: r[c] = df[c].mean(); r[c + '_sd'] = df[c].std(ddof=1) if len(df) > 1 else 0.0
        rows.append(r)
    tab = pd.DataFrame(rows); tab.to_csv(f'{R}/table_pooled_models.csv', index=False)
    # skill vs persistence with CI
    sk = []
    for name in out:
        if name == 'Persistence': continue
        c = compare(out, meta, name, 'Persistence', 'pop'); c2 = compare(out, meta, name, 'Train-mean', 'end')
        sk.append(dict(model=name, pop_gain_vs_persistence=c['delta'], pop_lo=c['lo'], pop_hi=c['hi'], pop_sig=c['sig'],
                       end_gain_vs_trainmean=c2['delta'], end_lo=c2['lo'], end_hi=c2['hi'], end_sig=c2['sig']))
    pd.DataFrame(sk).to_csv(f'{R}/table_skill_vs_baselines.csv', index=False)
    # fusion comparison: LPCF (K=3) vs alternatives (same backbone)
    ref = 'LPCF (K=3)'; fus = []
    for alt in ['K=1 (learned fixed w)', 'Uniform fixed w', 'Early concat', 'TFT-style selection', 'LPCF (K=2)', 'LPCF (K=3) + phase sup.', 'LPCF (K=3, no phi->E)', 'TFT-lite', 'LSTM', 'DLinear', 'XGBoost', 'Random Forest', 'Persistence']:
        cp = compare(out, meta, ref, alt, 'pop'); ce = compare(out, meta, ref, alt, 'end')
        fus.append(dict(vs=alt, pop_delta_MAE=cp['delta'], pop_lo=cp['lo'], pop_hi=cp['hi'], pop_sig=cp['sig'], pop_pairwise_win=cp['pairwise_win_frac'],
                        end_delta_MAE=ce['delta'], end_lo=ce['lo'], end_hi=ce['hi'], end_sig=ce['sig'], end_pairwise_win=ce['pairwise_win_frac']))
    pd.DataFrame(fus).to_csv(f'{R}/table_lpcf_vs_alternatives.csv', index=False)
    # best-of-all pairwise: which model is best on each target and is it significantly better than runner-up
    # per-fold breakdown for the headline models
    pf = []
    for fi in sorted(set(meta['fold'])):
        m = meta['fold'] == fi
        for name in ['Persistence', 'LPCF (K=3)', 'K=1 (learned fixed w)', 'Early concat', 'LSTM', 'XGBoost']:
            e = abs_err_pop({k: v[m] if k not in ('X',) else v for k, v in meta.items() if k in ('pop_last', 'pop_true', 'sd_delta')}, out[name]['dpop'][:, m]) if False else \
                np.abs(meta['pop_last'][m][None] + out[name]['dpop'][:, m] * meta['sd_delta'][m][None] - meta['pop_true'][m][None])
            pf.append(dict(fold=int(fi), model=name, MAE_log=e.mean(), n=int(m.sum())))
    pd.DataFrame(pf).to_csv(f'{R}/table_per_fold.csv', index=False)
    json.dump(summary, open(f'{R}/summary.json', 'w'), indent=1, default=float)
    pickle.dump((out, meta), open(f'{R}/_pooled_stacked.pkl', 'wb'))
    print(tab[['model', 'MAE_log', 'MAE_log_sd', 'MASE', 'R2_delta', 'MAE_end_unc', 'Cindex']].round(3).to_string())
    print(pd.DataFrame(fus).round(3).to_string())

