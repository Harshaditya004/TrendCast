"""Rolling-origin experiments.  usage: python run_rolling.py <pooled|logo|signal_ablation> <out.pkl> [tau]
pooled: all games, chronological folds.  logo: leave-one-game-out within the same chronological folds (train games' targets < fold start).
signal_ablation: remove one input signal at a time (S=3) for LPCF / K=1 / LSTM."""
import sys, time, pickle, numpy as np, torch
sys.path.insert(0, '.')
import windows as W, models as M, baselines as B
torch.set_num_threads(1)
mode, out = sys.argv[1], sys.argv[2]; tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
P = W.Panel(tau=tau); CFG = W.CFG
SEEDS = CFG['seeds'] if mode in ('pooled', 'gate_lr') else CFG['seeds'][:5]
LAM_PHI = 0.3
NEUR_LOGO = ['LPCF (K=3)', 'K=1 (learned fixed w)', 'Uniform fixed w', 'Early concat', 'TFT-style selection', 'LSTM', 'DLinear', 'TFT-lite']
res = {}

def seeded_model(name, S, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return M.NEURAL[name](S)

def sel(d, cols): d = dict(d); d['X'] = d['X'][:, :, cols]; return d

def run_models(tr, te, va, names_neural, names_tab, cols=None, sup=True):
    cols = cols if cols is not None else [0, 1, 2, 3]
    trs, tes, vas = sel(tr, cols), sel(te, cols), sel(va, cols); S = len(cols); r = {}
    for name in names_neural:
        outs = []
        for sd in SEEDS:
            m = seeded_model(name, S, sd); M.fit(m, trs, sd, val=vas); outs.append(M.predict(m, tes))
        r[name] = {k: np.stack([o[k] for o in outs]) for k in outs[0]}
    if sup and 'LPCF (K=3)' in names_neural:
        outs = []
        for sd in SEEDS:
            m = seeded_model('LPCF (K=3)', S, sd); M.fit(m, trs, sd, lam_phi=LAM_PHI, val=vas); outs.append(M.predict(m, tes))
        r['LPCF (K=3) + phase sup.'] = {k: np.stack([o[k] for o in outs]) for k in outs[0]}
    for name in names_tab:
        outs = []
        for sd in (SEEDS if name in ('Random Forest', 'XGBoost', 'XGBoost-AFT') else SEEDS[:1]):
            trs_all = {k: np.concatenate([trs[k], vas[k]]) for k in trs}   # tabular models have no early stopping -> use val rows for training too (more data than the nets)
            outs.append(B.Tab(name, sd).fit(trs_all).predict(tes))
        r[name] = {k: np.stack([o[k] for o in outs]) for k in outs[0]}
    return r

t_start = time.time()
for fi, fold in enumerate(CFG['folds']):
    if mode == 'pooled':
        tr, te, st, t0, va, tra = W.split_fold(P, fold)
        res[fi] = dict(meta=te, res=run_models(tr, te, va, list(M.NEURAL), B.TAB_MODELS))
    elif mode == 'logo':
        res[fi] = {}
        for g in range(len(P.games)):
            others = [x for x in range(len(P.games)) if x != g]
            tr, _, st, t0, va, _ = W.split_fold(P, fold, train_games=others)
            _, te, _, _, _, _ = W.split_fold(P, fold, game_ids=[g], train_games=others)
            res[fi][g] = dict(meta=te, res=run_models(tr, te, va, NEUR_LOGO, ['Persistence', 'Ridge', 'Random Forest', 'XGBoost'], sup=False))
            print(fi, P.games[g], f'{time.time()-t_start:.0f}s', flush=True)
    elif mode == 'gate_lr':
        tr, te, st, t0, va, tra = W.split_fold(P, fold); res[fi] = dict(meta=te, res={})
        for name in ['LPCF (K=3)', 'K=1 (learned fixed w)']:
            for gm in [1.0, 10.0, 30.0]:
                outs = []
                for sd in SEEDS:
                    m = seeded_model(name, 4, sd); M.fit(m, tr, sd, val=va, gate_mult=gm); outs.append(M.predict(m, te))
                res[fi]['res'][(name, f'gate_lr_x{gm:g}')] = {k: np.stack([o[k] for o in outs]) for k in outs[0]}
    elif mode == 'signal_ablation':
        tr, te, st, t0, va, tra = W.split_fold(P, fold); res[fi] = dict(meta=te, res={})
        for drop in [None, 0, 1, 2, 3]:
            cols = [c for c in range(4) if c != drop]
            r = run_models(tr, te, va, ['LPCF (K=3)', 'K=1 (learned fixed w)', 'LSTM'], [], cols=cols, sup=False)
            for k, v in r.items(): res[fi]['res'][(k, 'all4' if drop is None else f'-{W.SIG4[drop]}')] = v
    print('fold', fi, 'done', f'{time.time()-t_start:.0f}s', flush=True)
    pickle.dump(dict(res=res, games=P.games, tau=tau, seeds=SEEDS), open(out, 'wb'))
