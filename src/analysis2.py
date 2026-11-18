"""Second analysis stage: LOGO, prequential online, signal ablation, phase validity, redundancy, coverage, tau sensitivity, figures."""
import sys, os, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import analysis as A, windows as W, data as D
R, P, H = A.R, A.P, W.H; FIG = f'{R}/figs'; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
exists = os.path.exists


# ------------------------------------------------------------- coverage + redundancy
def coverage_and_redundancy():
    rows, cor_rows = [], []
    allz = []
    for g, name in enumerate(P.games):
        p = pd.read_csv(f'{D.RAW}/{name}/players_daily.csv', parse_dates=['date']); r = pd.read_csv(f'{D.RAW}/{name}/reviews/reviews_daily.csv', parse_dates=['date'])
        full = pd.date_range(p.date.min(), p.date.max())
        rows.append(dict(game=name, player_days=len(p), start=str(p.date.min().date()), end=str(p.date.max().date()), calendar_gaps=len(full) - len(p),
                         review_days=len(r), review_days_missing=len(full) - len(r.set_index('date').reindex(full).dropna()),
                         trends_days=len([1 for _ in []]) + (len(pd.read_csv(f'{D.RAW}/{name}/trends_raw.csv')) if exists(f'{D.RAW}/{name}/trends_raw.csv') else 0),
                         mean_players=p.player_peak.mean(), min_peak=p.player_peak.min(), max_peak=p.player_peak.max(),
                         peak_mean_daily_corr=p.player_peak.corr(p.player_mean), mean_over_peak=(p.player_mean / p.player_peak).mean()))
        z = (P.F[g][19:] - np.nanmean(P.F[g][19:], 0)); allz.append(z)
        cm = pd.DataFrame(P.F[g][19:], columns=W.SIG4).corr()
        for i in range(4):
            for j in range(i + 1, 4): cor_rows.append(dict(game=name, a=W.SIG4[i], b=W.SIG4[j], r=cm.iloc[i, j]))
    Z = np.concatenate(allz); Z = Z / Z.std(0)
    pooled = pd.DataFrame(Z, columns=W.SIG4).corr(); vif = np.diag(np.linalg.inv(pooled.values))
    pd.DataFrame(rows).to_csv(f'{R}/table_data_coverage.csv', index=False); pd.DataFrame(cor_rows).to_csv(f'{R}/table_signal_corr_per_game.csv', index=False)
    pooled.to_csv(f'{R}/table_signal_corr_pooled.csv'); pd.DataFrame(dict(signal=W.SIG4, VIF=vif)).to_csv(f'{R}/table_vif.csv', index=False)
    print(pooled.round(2)); print('VIF', dict(zip(W.SIG4, vif.round(2))))


# ------------------------------------------------------------- endurance targets
def tau_sensitivity():
    rows = []
    for tau in [0.3, 0.5, 0.7, 0.8, 0.9]:
        for g, name in enumerate(P.games):
            E, C = D.endurance(P.cnt[g], tau, W.KCAP); ok = ~np.isnan(E)
            rows.append(dict(tau=tau, game=name, n=int(ok.sum()), censored_frac=C[ok].mean(), at_cap_frac_uncens=((E == W.KCAP) & ~C)[ok].mean(), mean_E=np.nanmean(E),
                             uncensored_E_mean=E[ok & ~C].mean() if (ok & ~C).any() else np.nan, uncensored_E_sd=E[ok & ~C].std() if (ok & ~C).any() else np.nan))
    pd.DataFrame(rows).to_csv(f'{R}/table_tau_labels.csv', index=False)
    # log-scale reading of the definition is degenerate
    g = 0; Pl = np.log1p(P.cnt[g]); print('log-scale: 0.3*max P =', round(0.3 * np.nanmax(Pl), 2), 'vs min P', round(np.nanmin(Pl), 2))
    # leakage magnitude: how many training windows change label if labels were built from the full series instead of truncated at the cut-off
    lk = []
    for fi, fold in enumerate(W.CFG['folds']):
        t0 = fold[0] - H; ch = tot = 0
        for g in range(len(P.games)):
            Et, Ct = P.labels_known_at(g, t0)
            for t in range(19, t0 - H + 1):
                tot += 1; ch += (Et[t] != P.E_full[g][t])
        lk.append(dict(fold=fi, train_windows=tot, labels_that_would_leak=ch, frac=ch / tot))
    pd.DataFrame(lk).to_csv(f'{R}/table_label_leakage_check.csv', index=False); print(pd.DataFrame(lk))


# ------------------------------------------------------------- phase validity
def hindsight_labels(meta):
    lab = []
    for g, t in zip(meta['g'], meta['t']):
        c = P.cnt[g]; gr = c[min(t + 7, P.T - 1)] / c[t - 7] - 1; M = np.nanmax(c[:t + 8])
        lab.append(0 if gr >= 0.05 else (2 if (gr <= -0.05 and c[t] < 0.9 * M) else 1))
    return np.array(lab)


def best_perm_acc(y, z, K=3):
    C = np.zeros((K, K)); [C.__setitem__((a, b), C[a, b] + 1) for a, b in zip(y, z)]
    r, c = linear_sum_assignment(-C); return C[r, c].sum() / len(y)


def js(p, q):
    m = 0.5 * (p + q); kl = lambda a, b: (a * np.log((a + 1e-12) / (b + 1e-12))).sum(); return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def phase_validity(out, meta):
    weak, hind = meta['phase'], hindsight_labels(meta); rows = []
    rng = np.random.default_rng(0)
    for mname in ['LPCF (K=3)', 'LPCF (K=3) + phase sup.']:
        Phi = out[mname]['phi']; Wt = out[mname]['w']
        for s in range(Phi.shape[0]):
            ph = Phi[s]; am = ph.argmax(1); ent = -(ph * np.log(ph + 1e-12)).sum(1)
            # exact A recovery: w = phi A (least squares)
            A_est = np.linalg.lstsq(ph, Wt[s], rcond=None)[0]; Ak = np.clip(A_est, 1e-9, None); Ak = Ak / Ak.sum(1, keepdims=True)
            jsd = np.mean([js(Ak[i], Ak[j]) for i in range(Ak.shape[0]) for j in range(i + 1, Ak.shape[0])])
            rows.append(dict(model=mname, seed=s, nmi_weak=normalized_mutual_info_score(weak, am), nmi_hind=normalized_mutual_info_score(hind, am),
                             ari_weak=adjusted_rand_score(weak, am), acc_weak_bestperm=best_perm_acc(weak, am), acc_hind_bestperm=best_perm_acc(hind, am),
                             nmi_shuffled=normalized_mutual_info_score(weak, rng.permutation(am)), mean_entropy=ent.mean(), max_entropy=np.log(ph.shape[1]),
                             frac_conf_gt90=(ph.max(1) > 0.9).mean(), n_phases_used=len(np.unique(am)), mean_pairwise_JS_signal_weights=jsd,
                             w_std_across_samples=Wt[s].std(0).mean(), rank_phi=int(np.linalg.matrix_rank(ph, tol=1e-3))))
    ph_df = pd.DataFrame(rows); ph_df.groupby('model').mean(numeric_only=True).drop(columns='seed').to_csv(f'{R}/table_phase_validity.csv')
    print(ph_df.groupby('model').mean(numeric_only=True).drop(columns='seed').round(3).T)
    # weak vs hindsight agreement (label definitions themselves)
    agree = (weak == hind).mean(); print('weak(past-only) vs hindsight label agreement', round(agree, 3), 'NMI', round(normalized_mutual_info_score(weak, hind), 3))
    pd.DataFrame([dict(agreement=agree, nmi=normalized_mutual_info_score(weak, hind), weak_dist=str(np.bincount(weak, minlength=3).tolist()), hind_dist=str(np.bincount(hind, minlength=3).tolist()))]).to_csv(f'{R}/table_weak_vs_hindsight.csv', index=False)
    # phase-conditional effective weights for the unsupervised LPCF (seed-avg)
    Phi = out['LPCF (K=3)']['phi'].mean(0); Wt = out['LPCF (K=3)']['w'].mean(0)
    cond = pd.DataFrame([dict(group=f'weak={n}', **{W.SIG4[j]: Wt[weak == k, j].mean() for j in range(4)}, n=int((weak == k).sum())) for k, n in enumerate(['rising', 'peak', 'decay'])])
    cond.to_csv(f'{R}/table_weights_by_weak_phase.csv', index=False); print(cond.round(3))
    # figure: phase probabilities for each game over test days (seed-avg, unsup) + weak labels
    fig, ax = plt.subplots(2, 5, figsize=(14, 4.6), sharex=True)
    for g, name in enumerate(P.games):
        m = meta['g'] == g; o = np.argsort(meta['t'][m]); t = meta['t'][m][o]
        ax[0, g].stackplot(t, Phi[m][o].T, colors=['#2a9d8f', '#e9c46a', '#e76f51'], labels=['phase 1', 'phase 2', 'phase 3']); ax[0, g].set_title(name, fontsize=9); ax[0, g].set_ylim(0, 1)
        ax[1, g].scatter(t, meta['phase'][m][o], c=['#2a9d8f', '#e9c46a', '#e76f51'][0] if False else np.array(['#2a9d8f', '#e9c46a', '#e76f51'])[meta['phase'][m][o]], s=10)
        ax[1, g].set_yticks([0, 1, 2]); ax[1, g].set_yticklabels(['rising', 'peak', 'decay']); ax[1, g].set_xlabel('origin day')
    ax[0, 0].set_ylabel('LPCF φ (seed mean)'); ax[1, 0].set_ylabel('weak label'); ax[0, 0].legend(fontsize=7, loc='lower left'); fig.tight_layout(); fig.savefig(f'{FIG}/phase_probabilities.png', dpi=160); plt.close()


def long_history_lifecycle():
    """Independent lifecycle check on the full monthly history (2015-2026): weak rules vs visible rise/peak/decay."""
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.2)); rows = []
    for g, name in enumerate(P.games):
        raw = pd.DataFrame(json.load(open(f'{D.RAW}/{name}/players_raw.json')), columns=['t', 'v']); raw['t'] = pd.to_datetime(raw.t, unit='ms')
        m = raw[(raw.t.dt.day == 1) & (raw.t.dt.hour == 0) & (raw.t < '2026-07-01')].set_index('t').v.astype(float)
        lab = []
        for i in range(3, len(m)):
            gr = m.iloc[i] / m.iloc[i - 3] - 1; M = m.iloc[:i + 1].max()
            lab.append(0 if gr >= 0.05 else (2 if (m.iloc[i] < 0.9 * M and gr < 0) else 1))
        lab = np.array(lab); idx = m.index[3:]
        ax[g].semilogy(m.index, m.values, color='k', lw=1)
        for k, c in enumerate(['#2a9d8f', '#e9c46a', '#e76f51']):
            sel = lab == k; ax[g].scatter(idx[sel], m.values[3:][sel], s=6, c=c)
        ax[g].set_title(name, fontsize=9); ax[g].tick_params(axis='x', labelrotation=45)
        pk = m.idxmax(); pre = lab[(idx < pk)]; post = lab[(idx > pk)]
        rows.append(dict(game=name, months=len(m), peak_month=str(pk.date()), frac_rising_before_peak=(pre == 0).mean() if len(pre) else np.nan,
                         frac_decay_after_peak=(post == 2).mean() if len(post) else np.nan, frac_rising_after_peak=(post == 0).mean() if len(post) else np.nan,
                         daily_window_months_equiv='see manuscript: 89-day window sits in the late/plateau part of every lifecycle'))
    fig.tight_layout(); fig.savefig(f'{FIG}/long_history_lifecycle.png', dpi=160); plt.close()
    pd.DataFrame(rows).to_csv(f'{R}/table_long_history_phases.csv', index=False); print(pd.DataFrame(rows).iloc[:, :6].round(2))


# ------------------------------------------------------------- LOGO
def logo():
    d = pickle.load(open(f'{R}/rolling_logo.pkl', 'rb')); folds = sorted(d['res']); G = len(P.games)
    if len(folds) < 3: print('logo incomplete'); return
    names = list(d['res'][0][0]['res'].keys()); rows = []; stacked = {}
    meta_all = None
    for name in names:
        dp, en = [], []
        for g in range(G):
            dp.append(np.concatenate([d['res'][fi][g]['res'][name]['dpop'] for fi in folds], 1)); en.append(np.concatenate([d['res'][fi][g]['res'][name]['end'] for fi in folds], 1))
        n_s = max(x.shape[0] for x in dp)
        dp = [np.repeat(x, n_s // x.shape[0], 0) if x.shape[0] != n_s else x for x in dp]; en = [np.repeat(x, n_s // x.shape[0], 0) if x.shape[0] != n_s else x for x in en]
        stacked[name] = dict(dpop=np.concatenate(dp, 1), end=np.concatenate(en, 1))
    metas = [{k: np.concatenate([d['res'][fi][g]['meta'][k] for fi in folds]) for k in d['res'][0][g]['meta'] if k != 'X'} for g in range(G)]
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}
    sds = {fi: W.norm_stats(P, W.CFG['folds'][fi][0] - H)[2] for fi in folds}
    foldid = np.concatenate([np.concatenate([np.full(len(d['res'][fi][g]['meta']['t']), fi) for fi in folds]) for g in range(G)])
    meta['sd_delta'] = np.array([sds[f] for f in foldid]); meta['mase'] = np.array([A.mase_scale(W.CFG['folds'][f][0] - H) for f in foldid])
    for name in names:
        df = A.per_seed_metrics(stacked, meta, name); r = dict(model=name)
        for c in ['MAE_log', 'MASE', 'R2_delta', 'MAE_end_unc', 'Cindex']: r[c] = df[c].mean(); r[c + '_sd'] = df[c].std(ddof=1) if len(df) > 1 else 0
        for g, gname in enumerate(P.games):
            m = meta['g'] == g; r[f'MAE_{gname}'] = np.abs(meta['pop_last'][m][None] + stacked[name]['dpop'][:, m] * meta['sd_delta'][m][None] - meta['pop_true'][m][None]).mean()
        rows.append(r)
    pd.DataFrame(rows).to_csv(f'{R}/table_logo.csv', index=False); print(pd.DataFrame(rows)[['model', 'MAE_log', 'MAE_log_sd', 'MASE', 'MAE_end_unc', 'Cindex']].round(3).to_string())
    cmp_rows = []
    for alt in ['Persistence', 'K=1 (learned fixed w)', 'Uniform fixed w', 'Early concat', 'TFT-style selection', 'XGBoost', 'LSTM']:
        c = A.compare(stacked, meta, 'LPCF (K=3)', alt, 'pop'); ce = A.compare(stacked, meta, 'LPCF (K=3)', alt, 'end')
        cmp_rows.append(dict(vs=alt, pop_delta=c['delta'], lo=c['lo'], hi=c['hi'], sig=c['sig'], end_delta=ce['delta'], end_lo=ce['lo'], end_hi=ce['hi'], end_sig=ce['sig']))
    pd.DataFrame(cmp_rows).to_csv(f'{R}/table_logo_lpcf_vs.csv', index=False); print(pd.DataFrame(cmp_rows).round(3).to_string())


# ------------------------------------------------------------- online
def online():
    p = f'{R}/online.pkl'
    if not exists(p): return
    d = pickle.load(open(p, 'rb')); res, meta = d['res'], d['meta']
    sd = W.norm_stats(P, d['C0'])[2]; meta = dict(meta); meta['sd_delta'] = np.full(len(meta['t']), sd); meta['mase'] = np.full(len(meta['t']), A.mase_scale(d['C0']))
    rows = []
    for k, v in res.items():
        df = A.per_seed_metrics({k: v}, meta, k); r = dict(model=k, n_seeds=len(df))
        for c in ['MAE_log', 'MASE', 'R2_delta', 'MAE_end_unc', 'Cindex']: r[c] = df[c].mean(); r[c + '_sd'] = df[c].std(ddof=1) if len(df) > 1 else 0
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(f'{R}/table_online.csv', index=False); print(t.round(3).to_string())
    base = 'LPCF (K=3) [FT]'; cr = []
    for alt in res:
        if alt == base: continue
        c = A.compare(res, meta, base, alt, 'pop'); ce = A.compare(res, meta, base, alt, 'end')
        cr.append(dict(vs=alt, pop_delta=c['delta'], lo=c['lo'], hi=c['hi'], sig=c['sig'], end_delta=ce['delta'], end_lo=ce['lo'], end_hi=ce['hi'], end_sig=ce['sig']))
    pd.DataFrame(cr).to_csv(f'{R}/table_online_lpcf_ft_vs.csv', index=False)
    # effect of adaptation within each model family: [adaptive] vs [static]
    ar = []
    for fam in sorted({k.rsplit(' [', 1)[0] for k in res}):
        st = f'{fam} [static]'
        for v in ['FT', 'retrain', 'refit']:
            k = f'{fam} [{v}]'
            if k in res and st in res:
                c = A.compare(res, meta, k, st, 'pop'); ar.append(dict(model=fam, update=v, gain_vs_static=c['delta'], lo=c['lo'], hi=c['hi'], sig=c['sig']))
    pd.DataFrame(ar).to_csv(f'{R}/table_online_adaptation_effect.csv', index=False); print(pd.DataFrame(ar).round(3).to_string())
    # cumulative-error figure
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); days = np.unique(meta['t'])
    for k, c in [('Persistence [static]', 'k'), ('LPCF (K=3) [static]', '#e76f51'), ('LPCF (K=3) [FT]', '#e76f51'), ('K=1 (learned fixed w) [FT]', '#2a9d8f'), ('Early concat [FT]', '#264653'), ('XGBoost [refit]', '#8d99ae')]:
        if k not in res: continue
        e = A.abs_err_pop(meta, res[k]['dpop']).mean(0); cum = [e[meta['t'] <= dd].mean() for dd in days]
        ax.plot(days + H, cum, label=k, color=c, ls='--' if 'static' in k else '-')
    ax.set_xlabel('target day'); ax.set_ylabel('running prequential MAE (log scale)'); ax.legend(fontsize=6.5); fig.tight_layout(); fig.savefig(f'{FIG}/prequential_running_mae.png', dpi=160); plt.close()


def signal_ablation():
    p = f'{R}/signal_ablation.pkl'
    if not exists(p): return
    d = pickle.load(open(p, 'rb')); folds = sorted(d['res'])
    if len(folds) < 3: print('ablation incomplete'); return
    keys = list(d['res'][0]['res'].keys())
    out = {k: {q: np.concatenate([d['res'][fi]['res'][k][q] for fi in folds], 1) for q in ('dpop', 'end')} for k in keys}
    meta = {k: np.concatenate([d['res'][fi]['meta'][k] for fi in folds]) for k in d['res'][0]['meta'] if k != 'X'}
    sds = [W.norm_stats(P, W.CFG['folds'][fi][0] - H)[2] for fi in folds]; ms = [A.mase_scale(W.CFG['folds'][fi][0] - H) for fi in folds]
    meta['sd_delta'] = np.concatenate([np.full(len(d['res'][fi]['meta']['t']), sds[i]) for i, fi in enumerate(folds)]); meta['mase'] = np.concatenate([np.full(len(d['res'][fi]['meta']['t']), ms[i]) for i, fi in enumerate(folds)])
    rows = []
    for (m, cfg) in keys:
        df = A.per_seed_metrics({(m, cfg): out[(m, cfg)]}, meta, (m, cfg))
        r = dict(model=m, inputs=cfg, MAE_log=df.MAE_log.mean(), MAE_log_sd=df.MAE_log.std(ddof=1), MAE_end_unc=df.MAE_end_unc.mean(), Cindex=df.Cindex.mean())
        if cfg != 'all4':
            c = A.compare(out, meta, (m, 'all4'), (m, cfg), 'pop'); r.update(pop_removal_cost=c['delta'], lo=c['lo'], hi=c['hi'], sig=c['sig'])   # >0: removing the signal hurts
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(f'{R}/table_signal_ablation.csv', index=False); print(t.round(3).to_string())


if __name__ == '__main__':
    what = sys.argv[1:] or ['cov', 'tau', 'phase', 'long', 'logo', 'online', 'abl']
    if 'cov' in what: coverage_and_redundancy()
    if 'tau' in what: tau_sensitivity()
    if 'phase' in what:
        out, meta = pickle.load(open(f'{R}/_pooled_stacked.pkl', 'rb')); phase_validity(out, meta)
    if 'long' in what: long_history_lifecycle()
    if 'logo' in what: logo()
    if 'online' in what: online()
    if 'abl' in what: signal_ablation()
