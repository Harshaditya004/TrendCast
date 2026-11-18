"""Prequential (test-then-train) evaluation over the chronological stream with EQUAL new data for every adaptive method.
usage: python run_online.py <out.pkl>
Stream: origins t=C0..T-H-1.  At each origin day every model forecasts P(t+7) for all games BEFORE seeing its label.
Every U days each adaptive model receives exactly the same newly resolved windows (origin<=t-H; endurance labels re-derived from the series truncated at t).
 static  : trained once on data known at C0
 refit   : tabular models re-trained on everything revealed so far
 FT      : neural fine-tune (warm start) on the most recent buffer, 5 epochs, lr 1e-3
 retrain : neural re-trained from scratch on everything revealed (early stopping on the newest 7 origin days) - strongest equal-data reference
"""
import sys, pickle, time, copy, numpy as np, torch
sys.path.insert(0, '.')
import windows as W, models as M, baselines as B
torch.set_num_threads(1)
out = sys.argv[1]; P = W.Panel(tau=0.8); CFG = W.CFG; H = W.H; L = W.L
C0 = 48; U = 7; SEEDS = CFG['seeds'][:5]; FT_EPOCHS, FT_LR, BUF = 5, 1e-3, 64
NEUR = ['LPCF (K=3)', 'K=1 (learned fixed w)', 'Early concat', 'TFT-style selection', 'LSTM', 'DLinear', 'TFT-lite']
TAB = ['Ridge', 'Random Forest', 'XGBoost']
stats = W.norm_stats(P, C0)
origins = list(range(C0, P.T - H))
upd_days = [t for t in origins if t > C0 and (t - C0) % U == 0]
test_all = W.make_samples(P, origins, C0, stats, train=False)           # labels (full series) used only for scoring

def revealed(day):
    d = W.make_samples(P, range(6 + L - 1, day - H + 1), day, stats, train=True)
    o = np.argsort(d['t'], kind='stable'); return {k: v[o] for k, v in d.items()}
def split_val(d, day):
    vm = d['t'] > (day - H - 7); f = lambda m: {k: v[m] for k, v in d.items()}; return f(~vm), f(vm)
def sub(d, m): return {k: v[m] for k, v in d.items()}

res = {}
def record(name, seed, day_mask_preds):
    res.setdefault(name, {}).setdefault('dpop', {})[seed] = day_mask_preds['dpop']; res[name].setdefault('end', {})[seed] = day_mask_preds['end']

t_start = time.time()
init = revealed(C0); tr0, va0 = split_val(init, C0)
tday = test_all['t']
for name in NEUR + TAB + ['Persistence']:
    for sd in (SEEDS if name != 'Persistence' else [0]):
        if name in NEUR:
            variants = {'static': None, 'FT': None, 'retrain': None}
            base = M.NEURAL[name](4); M.fit(base, tr0, sd, val=va0)
            models = {'static': base, 'FT': copy.deepcopy(base), 'retrain': copy.deepcopy(base)}
        elif name in TAB:
            models = {'static': B.Tab(name, sd).fit(init), 'refit': B.Tab(name, sd).fit(init)}
        else:
            models = {'static': B.Tab(name, sd).fit(init)}
        preds = {v: dict(dpop=np.zeros(len(tday)), end=np.zeros(len(tday))) for v in models}
        for t in origins:
            if t in upd_days:                                           # label revelation + update (same data for every adaptive model)
                rv = revealed(t)
                if 'FT' in models:
                    buf = {k: v[-BUF:] for k, v in rv.items()}
                    M.fit(models['FT'], buf, sd, epochs=FT_EPOCHS, lr=FT_LR)
                if 'retrain' in models:
                    a, b = split_val(rv, t); m = M.NEURAL[name](4); M.fit(m, a, sd, val=b); models['retrain'] = m
                if 'refit' in models: models['refit'] = B.Tab(name, sd).fit(rv)
            ix = np.where(tday == t)[0]; d = sub(test_all, np.isin(np.arange(len(tday)), ix))
            for v, m in models.items():
                p = M.predict(m, d) if name in NEUR else m.predict(d)
                preds[v]['dpop'][ix] = p['dpop']; preds[v]['end'][ix] = p['end']
        for v in models: record(f'{name} [{v}]', sd, preds[v])
    print(name, f'{time.time()-t_start:.0f}s', flush=True)
    pickle.dump(dict(res={k: {q: np.stack([res[k][q][s] for s in sorted(res[k][q])]) for q in ('dpop', 'end')} for k in res}, meta=test_all, games=P.games, C0=C0, U=U, upd_days=upd_days), open(out, 'wb'))
