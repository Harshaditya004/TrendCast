"""Headline figures (light surface; blue = LPCF, orange = fusion alternatives, grey = references; labels are direct, grid recessive)."""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'); F = f'{R}/figs'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'text.color': '#0b0b0b', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
BL, OR, GR = '#2a78d6', '#eb6834', '#8d8c86'
t = pd.read_csv(f'{R}/table_pooled_models.csv').set_index('model')
order = t.sort_values('MAE_log').index.tolist()
col = lambda m: BL if m.startswith('LPCF') else (OR if m in ('K=1 (learned fixed w)', 'Uniform fixed w', 'Early concat', 'TFT-style selection') else GR)
fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
y = np.arange(len(order))
ax[0].barh(y, t.loc[order, 'MAE_log'], xerr=t.loc[order, 'MAE_log_sd'], color=[col(m) for m in order], height=0.62, error_kw=dict(ecolor='#52514e', lw=1))
ax[0].set_yticks(y); ax[0].set_yticklabels(order); ax[0].invert_yaxis(); ax[0].axvline(t.loc['Persistence', 'MAE_log'], color='#0b0b0b', lw=1, ls=':')
ax[0].set_xlabel('7-day-ahead popularity MAE, log scale (lower is better)\nmean ± sd over seeds; dotted = persistence'); ax[0].set_title('Popularity, rolling-origin, 5 real games', loc='left', fontsize=10)
o2 = t.sort_values('Cindex', ascending=False).index.tolist(); y2 = np.arange(len(o2))
ax[1].barh(y2, t.loc[o2, 'Cindex'], color=[col(m) for m in o2], height=0.62); ax[1].axvline(0.5, color='#0b0b0b', lw=1, ls=':')
ax[1].set_yticks(y2); ax[1].set_yticklabels(o2); ax[1].invert_yaxis(); ax[1].set_xlim(0.3, 0.85)
ax[1].set_xlabel('Endurance concordance index, censoring-aware\n(higher is better; dotted = chance)'); ax[1].set_title('Endurance (τ = 0.8)', loc='left', fontsize=10)
fig.tight_layout(); fig.savefig(f'{F}/main_results.png', dpi=170); plt.close()

d = pd.read_csv(f'{R}/table_lpcf_vs_alternatives.csv')
d = d[d.vs.isin(['K=1 (learned fixed w)', 'Uniform fixed w', 'Early concat', 'TFT-style selection', 'LPCF (K=2)', 'LPCF (K=3) + phase sup.', 'LPCF (K=3, no phi->E)'])]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True); y = np.arange(len(d))
for a, (c, lo, hi, ttl) in zip(ax, [('pop_delta_MAE', 'pop_lo', 'pop_hi', 'Popularity: Δ MAE (log)'), ('end_delta_MAE', 'end_lo', 'end_hi', 'Endurance: Δ MAE (days)')]):
    a.errorbar(d[c], y, xerr=[d[c] - d[lo], d[hi] - d[c]], fmt='o', color=BL, ecolor=BL, capsize=3, ms=5); a.axvline(0, color='#0b0b0b', lw=1)
    a.set_xlabel(ttl + '\n>0: LPCF better · 95% block-bootstrap CI'); a.grid(axis='x', color='#e6e5e1', lw=0.6)
ax[0].set_yticks(y); ax[0].set_yticklabels('vs ' + d.vs); ax[0].invert_yaxis(); fig.tight_layout(); fig.savefig(f'{F}/lpcf_vs_alternatives.png', dpi=170); plt.close()
