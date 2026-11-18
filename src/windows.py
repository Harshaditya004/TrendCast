"""Window construction with leak-free normalisation and cut-off-aware endurance labels."""
import numpy as np
from data import CFG, load_game, features, endurance

SIG4 = ['popularity', 'engagement', 'rating', 'review_growth']   # the four signals that exist in the submitted archive
L, H, KCAP = CFG['L'], CFG['h'], CFG['K_cap']


class Panel:
    """All games, raw (unnormalised) causal signals on a common day index 0..T-1."""
    def __init__(self, games=None, tau=0.8):
        self.games = games or CFG['games']
        self.tau = tau
        self.F, self.cnt, self.raw_new_rev = [], [], []
        for g in self.games:
            df = load_game(g)
            df['new_rev'] = df['new_rev'].fillna(0.0); df['pos'] = df['pos'].fillna(0.0); df['tot'] = df['tot'].fillna(0.0)
            f = features(df)
            self.F.append(f[SIG4].values.astype(float)); self.cnt.append(f['cnt'].values.astype(float))
        self.T = min(len(c) for c in self.cnt)
        self.F = [a[:self.T] for a in self.F]; self.cnt = [c[:self.T] for c in self.cnt]
        # full-series endurance (used ONLY as test labels)
        self.E_full, self.C_full = [], []
        for c in self.cnt:
            e, ce = endurance(c, tau, KCAP); self.E_full.append(e); self.C_full.append(ce)

    def labels_known_at(self, g, day):
        """Endurance labels re-derived from the series truncated at `day` (no future info); censored if unresolved."""
        c = self.cnt[g][:day + 1]
        return endurance(c, self.tau, KCAP)


def weak_phase(cnt, t, delta=None, decay_frac=None, w=7):
    delta = CFG['delta_growth'] if delta is None else delta
    decay_frac = CFG['decay_frac'] if decay_frac is None else decay_frac
    g = cnt[t] / cnt[t - w] - 1.0
    M = np.nanmax(cnt[:t + 1])
    if g >= delta: return 0                      # rising
    if cnt[t] < decay_frac * M and g < 0: return 2   # decay
    return 1                                     # peak / plateau


def norm_stats(P, cutoff):
    """Training-period-only statistics: per-game mean, pooled within-game std per signal, pooled std of 7-day log change."""
    S = len(SIG4); means = []; devs = [[] for _ in range(S)]; dl = []
    for g in range(len(P.games)):
        seg = P.F[g][6:cutoff + 1]
        m = np.nanmean(seg, axis=0); means.append(m)
        for s in range(S):
            d = seg[:, s] - m[s]; devs[s].append(d[~np.isnan(d)])
        pop = P.F[g][:, 0]; dl.append((pop[6 + H:cutoff + 1] - pop[6:cutoff + 1 - H]))
    sd = np.array([max(np.concatenate(devs[s]).std(), 1e-3) for s in range(S)])
    sd_delta = max(np.concatenate(dl).std(), 1e-3)
    return np.array(means), sd, sd_delta


def make_samples(P, origins, cutoff, stats, game_ids=None, train=True):
    """Build samples for origin days in `origins` (per game). train=True -> labels from series truncated at `cutoff`."""
    means, sd, sd_delta = stats
    game_ids = range(len(P.games)) if game_ids is None else game_ids
    out = dict(X=[], g=[], t=[], dpop=[], pop_last=[], pop_true=[], end=[], cen=[], phase=[], cnt_t=[], cnt_target=[])
    for g in game_ids:
        if train: E, C = P.labels_known_at(g, cutoff)
        else: E, C = P.E_full[g], P.C_full[g]
        for t in origins:
            if t - L + 1 < 6 or t + H >= P.T: continue
            win = P.F[g][t - L + 1:t + 1].copy()
            win = (win - means[g]) / sd
            win = np.nan_to_num(win, nan=0.0)            # missing 7d rating -> game training mean (0 after centring)
            pop = P.F[g][:, 0]
            out['X'].append(win); out['g'].append(g); out['t'].append(t)
            out['dpop'].append((pop[t + H] - pop[t]) / sd_delta)
            out['pop_last'].append(pop[t]); out['pop_true'].append(pop[t + H])
            out['end'].append(E[t]); out['cen'].append(C[t])
            out['phase'].append(weak_phase(P.cnt[g], t))
            out['cnt_t'].append(P.cnt[g][t]); out['cnt_target'].append(P.cnt[g][t + H])
    d = {k: np.array(v) for k, v in out.items()}
    d['X'] = d['X'].astype(np.float32)
    return d


def split_fold(P, fold, game_ids=None, train_games=None):
    """Rolling-origin fold: test target days in [a,b). Training uses origins whose popularity target is known by test start."""
    a, b = fold
    t0 = a - H                                     # first test origin; data up to day t0 is the known history
    stats = norm_stats(P, t0)
    tr_all = make_samples(P, range(6 + L - 1, t0 - H + 1), t0, stats, game_ids=train_games, train=True)
    # chronological validation block = last 7 origin days whose popularity target is known at t0 (used for early stopping only)
    vm = tr_all['t'] > (t0 - H - 7)
    sub = lambda d, m: {k: v[m] for k, v in d.items()}
    tr, va = sub(tr_all, ~vm), sub(tr_all, vm)
    te = make_samples(P, range(t0, b - H), t0, stats, game_ids=game_ids, train=False)
    return tr, te, stats, t0, va, tr_all
