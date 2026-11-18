"""Real-data construction: aligned daily signals from the submitted Steam files.
Signals: popularity (log 7d trailing mean of daily PEAK players), engagement (log 7d trailing mean of daily MEAN players),
rating (positive/total, 7d pooled), review_growth (log1p daily new reviews, 7d trailing mean), search_interest (Google Trends; NOT in archive -> masked).
"""
import json, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, 'src', 'config.json')))
RAW = os.path.join(ROOT, 'data', 'raw')
SIGNALS = ['popularity', 'engagement', 'rating', 'review_growth', 'search_interest']

def load_game(g):
    p = pd.read_csv(f'{RAW}/{g}/players_daily.csv', parse_dates=['date']).set_index('date')
    r = pd.read_csv(f'{RAW}/{g}/reviews/reviews_daily.csv', parse_dates=['date']).set_index('date')
    idx = pd.date_range(p.index.min(), p.index.max(), freq='D')
    df = pd.DataFrame(index=idx)
    df['peak'] = p['player_peak'].reindex(idx)          # counts
    df['mean'] = p['player_mean'].reindex(idx)
    df['new_rev'] = r['review_growth'].reindex(idx)
    df['pos'] = r['positive_reviews'].reindex(idx)
    df['tot'] = r['total_reviews'].reindex(idx)
    df['review_observed'] = df['new_rev'].notna()        # missing days are marked, not imputed
    df['search'] = np.nan   # Google Trends export (data/raw/<game>/trends_raw.csv) is NOT in the submitted archive -> signal absent, not imputed
    return df

def features(df):
    """Trailing-7d smoothed log-scale signals (causal: uses only days <= t)."""
    f = pd.DataFrame(index=df.index)
    f['cnt'] = df['peak'].rolling(7, min_periods=7).mean()          # count-scale popularity (for endurance, MASE, count errors)
    f['popularity'] = np.log1p(f['cnt'])
    f['engagement'] = np.log1p(df['mean'].rolling(7, min_periods=7).mean())
    pos = df['pos'].rolling(7, min_periods=1).sum(); tot = df['tot'].rolling(7, min_periods=1).sum()
    f['rating'] = (pos / tot.replace(0, np.nan))
    f['review_growth'] = np.log1p(df['new_rev'].rolling(7, min_periods=1).mean())
    f['search_interest'] = np.nan
    f['review_missing'] = (~df['review_observed']).astype(float)
    return f

def endurance(cnt, tau, K=30, scale='count'):
    """E(t)=max k<=K: P(t+j)>=tau*M(t) for all j=1..k ; M(t)=running peak up to t (past only). Returns (E, censored)."""
    c = np.asarray(cnt, float); n = len(c)
    M = np.fmax.accumulate(np.nan_to_num(c, nan=-np.inf))
    E = np.full(n, np.nan); cen = np.zeros(n, bool)
    for t in range(n):
        if not np.isfinite(c[t]): continue
        k = 0; failed = False
        for j in range(1, K + 1):
            if t + j >= n: break
            if c[t + j] >= tau * M[t]: k = j
            else: failed = True; break
        E[t] = k
        cen[t] = (not failed) and (k < K)                  # series ended before the condition failed and before the cap
    return E, cen
