"""Non-neural baselines on flattened windows. All receive identical windows, labels and splits as the neural models."""
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from data import CFG

KCAP = CFG['K_cap']


def flat(d):
    X = d['X']; return np.concatenate([X.reshape(len(X), -1), X[:, -1] - X[:, -7], X[:, -1] - X[:, 0]], 1)


class Tab:
    """kind in {'Persistence','Drift','Linear Regression','Ridge','Random Forest','XGBoost','XGBoost-AFT','Train-mean'}"""
    def __init__(self, kind, seed=0): self.kind, self.seed = kind, seed

    def fit(self, tr):
        k = self.kind; self.mu_e = np.mean(tr['end']); self.med_e = np.median(tr['end'])
        if k in ('Persistence', 'Drift', 'Train-mean'): return self
        X = flat(tr); yp = tr['dpop']; ye = tr['end']
        if k == 'Linear Regression': mk = lambda: LinearRegression()
        elif k == 'Ridge': mk = lambda: Ridge(alpha=10.0)
        elif k == 'Random Forest': mk = lambda: RandomForestRegressor(200, min_samples_leaf=3, random_state=self.seed, n_jobs=1)
        else: mk = lambda: xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=self.seed, n_jobs=1, verbosity=0)
        if k == 'XGBoost-AFT':
            self.mp = None
            dm = xgb.DMatrix(X); lo = ye + 1.0; hi = np.where(tr['cen'], np.inf, ye + 1.0)
            dm.set_float_info('label_lower_bound', lo); dm.set_float_info('label_upper_bound', hi)
            self.aft = xgb.train(dict(objective='survival:aft', aft_loss_distribution='normal', aft_loss_distribution_scale=1.0,
                                      max_depth=3, learning_rate=0.05, subsample=0.8, seed=self.seed, nthread=1, verbosity=0), dm, 150)
            self.mp = XGBPop(self.seed).fit(X, yp); return self
        self.mp = mk().fit(X, yp); self.me = mk().fit(X, ye)       # censored labels enter as observed lower bounds (no censoring support)
        return self

    def predict(self, d):
        k = self.kind; n = len(d['X'])
        if k == 'Persistence': return dict(dpop=np.zeros(n), end=np.full(n, self.med_e))
        if k == 'Train-mean': return dict(dpop=np.zeros(n), end=np.full(n, self.mu_e))
        if k == 'Drift':
            x = d['X'][:, :, 0]; slope = (x[:, -1] - x[:, -8]) / 7.0                  # linear extrapolation of the last week (normalised units)
            return dict(dpop=slope * 7.0, end=np.full(n, self.med_e))
        X = flat(d)
        if k == 'XGBoost-AFT':
            e = np.exp(np.clip(self.aft.predict(xgb.DMatrix(X)), -5, 4)) - 1.0
            return dict(dpop=self.mp.predict(X), end=np.clip(e, 0, KCAP))
        return dict(dpop=self.mp.predict(X), end=np.clip(self.me.predict(X), 0, KCAP))


class XGBPop:
    def __init__(self, seed): self.m = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=0)
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X): return self.m.predict(X)


TAB_MODELS = ['Persistence', 'Drift', 'Train-mean', 'Linear Regression', 'Ridge', 'Random Forest', 'XGBoost', 'XGBoost-AFT']
