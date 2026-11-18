import os, pandas as pd, numpy as np
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f'{ROOT}/results'
rd = lambda f: pd.read_csv(f'{R}/{f}.csv')
doc = Document(); st = doc.styles['Normal']; st.font.name = 'Calibri'; st.font.size = Pt(10.5)
H1 = lambda t: doc.add_heading(t, 1); H2 = lambda t: doc.add_heading(t, 2)
def para(t, bold=False, italic=False):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic; return p
def bullets(items):
    for i in items: doc.add_paragraph(i, style='List Bullet')
def table(df, widths=None, fs=8.5):
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = 'Light Grid Accent 1'
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
    for r in t.rows:
        for c in r.cells:
            for p in c.paragraphs:
                for run in p.runs: run.font.size = Pt(fs)
    doc.add_paragraph()
def fig(f, w=6.4, cap=''):
    doc.add_picture(f'{R}/figs/{f}', width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if cap: p = para(cap, italic=True); p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading('TrendCast Review-03: real-data evidence for LPCF and CPAF-Net', 0)
para('G Harshaditya Reddy · 23BCE0762 · Foundations of Data Science project · all numbers are produced by the code in the accompanying package (run_all.sh)', italic=True)

H1('1. Summary of what the real data show')
para('LPCF was implemented and compared with simpler fusion mechanisms and strong baselines on the real Steam data, using chronological rolling-origin evaluation, 8 seeds and paired block-bootstrap confidence intervals. The primary experiment gives a negative answer: LPCF does not provide a statistically significant improvement over simpler fusion methods.', bold=True)
bullets([
- **Popularity:** Persistence has the lowest MAE (0.118). LPCF (K=3) has MAE 0.136, compared with 0.128 for K=1 learned fixed weights, 0.126 for DLinear, and 0.126 for XGBoost. LPCF does not consistently outperform simpler or established baselines.

- **Fusion comparisons:** LPCF performs better than TFT-style selection (MAE difference +0.017 in favour of LPCF; 95% CI [0.003, 0.043]). It performs worse than K=1 learned fixed weights and early concatenation in the paired comparisons. Comparisons against uniform weighting and several other baselines are not statistically conclusive.

- **Endurance:** LPCF's endurance MAE and ranking performance vary across model comparisons. Tree-based models show stronger ranking performance than LPCF in the pooled results. The available evidence does not establish a general endurance advantage for LPCF.

- **Online adaptation and sensitivity analyses:** These sections use separate experiment outputs that were not regenerated with the corrected model initialization. Treat their numerical findings as preliminary until those experiments are rerun and their tables and narrative are updated consistently.

- **Lifecycle phases and data limitations:** Phase validity must be interpreted using the regenerated phase table. The dataset contains five games and four usable signals; Google Trends data are absent, so cross-platform five-signal fusion cannot be evaluated. The findings should be presented as limited real-data evidence rather than proof of superiority.

 'Data limits: Google Trends data are absent, so the cross-platform five-signal claim cannot be tested; the real data contain 5 games over 89 days, all mature titles past their launch peak.'])

H1('2. Status against the Review-03 list')
rows = [
 ('P1', 'Complete real five-signal dataset', 'Not possible from the supplied files. Four signals are aligned and documented; Google Trends CSVs are missing and the workspace could not reach Google or Steam. Search interest is left absent, not imputed.'),
 ('P1', 'Implement LPCF', 'Done (src/models.py): K phases, phase-conditioned convex signal weights, phase passed to the endurance head, optional weak-label loss.'),
 ('P1', 'Real-data forecasting', 'Done: 3 chronological folds, 170 test windows for popularity, 135 fully resolved windows for endurance (τ=0.8).'),
 ('P1', 'Strong baselines', 'Persistence, drift, Ridge, Random Forest, XGBoost, XGBoost-AFT, LSTM, DLinear, TFT-lite. TFT-lite is my compact re-implementation, not the official TFT.'),
 ('P1', 'Fusion novelty test', 'Done: LPCF vs K=1, uniform weights, early concatenation, TFT-style selection, K=2, no φ→E, phase supervision.'),
 ('P1', 'Online adaptation, equal data', 'Done: prequential test-then-train; static, fine-tune and retrain-from-scratch variants for every neural model, refit for tabular models, all fed the same resolved windows.'),
 ('P1', 'Repeated experiments', 'Done: 8 seeds (pooled) and 5 seeds (LOGO, online, ablations); mean ± sd; paired 7-day block-bootstrap CIs.'),
 ('P1', 'Leave-one-game-out', 'Done, inside the same chronological folds.'),
 ('P2', 'Validate lifecycle phases', 'Done with the data available (Section 7).'),
 ('P2', 'Expand data coverage', 'Not done: needs new Steam downloads.'),
 ('P2', 'Validate endurance targets', 'Done: censoring, threshold sensitivity, leakage check (Section 8).'),
 ('P2', 'Reproducibility package', 'Done: code, config, checksums, logs, predictions.')]
table(pd.DataFrame(rows, columns=['Priority', 'Required work', 'Status']), fs=8.5)

H1('3. Data and signals actually used')
cov = rd('table_data_coverage')[['game', 'player_days', 'review_days', 'trends_days', 'mean_players', 'peak_mean_daily_corr']]
table(cov)
para('All five games have 89 daily player observations from 2026-07-09 to 2026-10-05 and no calendar gaps. Trails in the Sky SC has reviews on only 57 of 89 days; absent days are treated as zero-review days, which is an assumption. Trends files: 0 of 5.')
para('Signals (all causal, 7-day trailing): popularity = log(1 + mean of daily peak players); engagement = log(1 + mean of daily mean players); rating = pooled positive/total reviews; review growth = log(1 + mean new reviews). Normalisation statistics use only the history known at each forecast cut-off.')
cor = pd.read_csv(f'{R}/table_signal_corr_pooled.csv', index_col=0).round(2).reset_index().rename(columns={'index': 'signal'})
vif = rd('table_vif')
para('Redundancy. Daily peak and daily mean players correlate at 0.66–0.92 per game. After removing each game’s mean, popularity and engagement correlate at 0.71 and popularity and review growth at 0.75. Variance-inflation factors are 3.2 (popularity), 2.2 (engagement), 1.3 (rating) and 3.2 (review growth): the signals are correlated but not collinear. The signal-removal ablation (Section 6) shows that no individual signal measurably helps LPCF.')
table(cor)

H1('4. Protocol')
bullets([
 'Window 14 days, horizon 7 days. Target for popularity is the 7-day log change, reported as log-scale MAE, RMSE, MASE, WAPE/sMAPE on counts and R² of the change.',
 'Rolling origin: test targets in days [55,65), [65,75), [75,89); every model is trained only on windows whose targets are known at the fold start, with the last 7 origin days used for early stopping. All five games are pooled; the leave-one-game-out variant trains on four games and tests on the fifth within the same date ranges.',
 'Same windows, targets, normalisation, seeds, optimiser, early stopping and maximum epochs for all neural models. No per-model hyper-parameter search was run, so tuned versions of any model could change the ranking. Tabular models use default settings and also train on the early-stopping rows.',
 'Backbone shared by all fusion variants: one BiLSTM with temporal attention per signal (hidden 16); only the weights w_s differ. Early concatenation uses a single BiLSTM over all signals.',
 'Statistics: differences use seed-averaged absolute errors and a 7-day moving-block bootstrap over origin days (all games resampled together, 2000 repetitions). With about 17 independent weeks of data, intervals are wide and several comparisons are made, so isolated significant results should be read cautiously.'])

H1('5. Popularity and endurance results (pooled, chronological)')
fig('main_results.png', 6.6, 'Figure 1. Left: popularity MAE (mean ± sd over seeds). Right: endurance C-index. Blue = LPCF variants, orange = alternative fusions, grey = other baselines.')
t = rd('table_pooled_models')
sel = t[['model', 'n_seeds', 'MAE_log', 'MAE_log_sd', 'MASE', 'R2_delta', 'WAPE_cnt', 'MAE_end_unc', 'Cindex']].copy()
sel.columns = ['Model', 'Seeds', 'MAE (log)', 'sd', 'MASE', 'R² of Δ', 'WAPE', 'Endurance MAE (days)', 'C-index']
table(sel.sort_values('MAE (log)'), fs=7.5)
para('MASE above 1 means worse than the in-sample persistence error; every model has MASE of at least 1.7 because test-period changes are larger than training-period changes (the last fold contains a large shock, persistence MAE 0.195 there vs 0.051 and 0.076 in the earlier folds). Constant predictors (persistence, train-mean) have a nominal C-index of 0.5; the table value differs slightly because their constants change by fold.')
sk = rd('table_skill_vs_baselines')[['model', 'pop_gain_vs_persistence', 'pop_lo', 'pop_hi', 'end_gain_vs_trainmean', 'end_lo', 'end_hi']]
sk.columns = ['Model', 'Pop: persistence MAE − model MAE', 'lo', 'hi', 'End: train-mean MAE − model MAE (days)', 'lo ', 'hi ']
para('Gain relative to naive references (positive = better, 95% CI):', bold=True); table(sk, fs=7.5)

H1('6. Does LPCF beat simpler fusion?')
fig('lpcf_vs_alternatives.png', 6.2, 'Figure 2. Difference in MAE between each alternative and LPCF (K=3); positive means LPCF is better. Default training setup.')
f = rd('table_lpcf_vs_alternatives')[['vs', 'pop_delta_MAE', 'pop_lo', 'pop_hi', 'pop_pairwise_win', 'end_delta_MAE', 'end_lo', 'end_hi']]
f.columns = ['LPCF vs', 'Pop Δ', 'lo', 'hi', 'Seed-pair win rate', 'End Δ (days)', 'lo ', 'hi ']
table(f, fs=7.5)
g = rd('table_gate_lr_sensitivity')[['model', 'setting', 'MAE_log', 'sd', 'w_std_across_samples', 'w_range_across_signals']]
g.columns = ['Model', 'Gate learning rate', 'MAE (log)', 'sd', 'Spread of w across samples', 'Range of mean w over signals']
para('Gate-training sensitivity. With the default learning rate the signal logits barely move (mean weights stay within 0.07 of uniform). A 10x or 30x multiplier gives learned, unequal weights and removes the LPCF deficit, but not in a way that favours LPCF over K=1:', bold=False); table(g, fs=8)
a = rd('table_signal_ablation')[['model', 'inputs', 'MAE_log', 'MAE_log_sd', 'pop_removal_cost', 'lo', 'hi']].copy(); a.columns = ['Model', 'Inputs', 'MAE (log)', 'sd', 'Cost of removing signal', 'lo', 'hi']
para('Signal-removal ablation (positive cost = removing the signal hurts). No removal measurably hurts LPCF; for K=1 and LSTM several removals slightly help.', bold=False); table(a, fs=7.5)

H1('7. Lifecycle-phase validity')
pv = rd('table_phase_validity') if False else pd.read_csv(f'{R}/table_phase_validity.csv')
pv2 = pv[['model', 'nmi_weak', 'nmi_shuffled', 'acc_weak_bestperm', 'nmi_hind', 'acc_hind_bestperm', 'mean_entropy', 'n_phases_used', 'mean_pairwise_JS_signal_weights']]
pv2.columns = ['Model', 'NMI vs weak labels', 'NMI shuffled', 'Accuracy vs weak (best mapping)', 'NMI vs hindsight', 'Accuracy vs hindsight', 'Mean entropy (max 1.10)', 'Phases used', 'JS between phase weight vectors']
table(pv2, fs=7.5)
para('Weak labels use only past data (growth over 7 days ≥ 3% = rising; below 90% of the running peak and falling = decay; otherwise peak). Hindsight labels use a centred ±7-day window and serve as an independent check. The two labelings agree on 68% of windows (NMI 0.26). Unsupervised LPCF phases are above chance but weak, the phase-specific signal weights are nearly identical (so LPCF behaves like fixed weighting), and the phase-probability curves change at fold boundaries because models are retrained there (Figure 3). I did not re-run the phase analysis for the 10x gate-learning-rate variant.')
fig('phase_probabilities.png', 6.6, 'Figure 3. LPCF phase probabilities (seed mean) and weak labels over the test days for each game.')
lh = pd.read_csv(f'{R}/table_long_history_phases.csv')[['game', 'months', 'peak_month', 'frac_rising_before_peak', 'frac_decay_after_peak', 'frac_rising_after_peak']]
para('The 89-day window sits in the plateau or decay part of lifecycles that began years earlier; no game in it has a genuine launch rise. The monthly history in players_raw.json shows this (weak-label rules applied to monthly data, 3-month growth):', bold=False); table(lh, fs=8)
fig('long_history_lifecycle.png', 6.6, 'Figure 4. Monthly player history with weak lifecycle labels (green rising, yellow peak, red decay).')

H1('8. Endurance target: censoring, threshold and leakage')
tl = rd('table_tau_labels').groupby('tau').agg(censored=('censored_frac', 'mean'), at_cap=('at_cap_frac_uncens', 'mean'), mean_E=('mean_E', 'mean')).reset_index()
tl.columns = ['τ', 'Censored share', 'At 30-day cap (uncensored)', 'Mean endurance (days)']
para('The definition as written compares P(t+j) ≥ τ·M(t) using the log-popularity P. That is degenerate: with τ=0.3 the threshold is about 2.9 while log-popularity never falls below 9.5, so every window reaches the cap. I therefore apply the threshold to the 7-day mean player count. Even then τ=0.3 and 0.5 produce only capped or censored windows. I chose τ=0.8 as the primary value from the label distribution alone (it is the first value at which endurance is neither capped nor censored for most windows and still varies across games), before looking at model results; at τ=0.9 several games have endurance near zero; τ=0.7 and 0.9 are sensitivity checks.')
table(tl)
bullets([
 'M(t) is the running peak of the observed daily series, which starts on 2026-07-09. The true all-time peak is usually years earlier, so endurance is relative to the 89-day-window peak (left truncation).',
 'Windows whose condition has not failed by the last observed day are marked censored. Training windows are labelled from the series truncated at the fold cut-off, so many are censored there; the neural models use an exact loss for observed endurance and a one-sided hinge for censored lower bounds. Tabular models cannot take censoring and receive the lower bound as the label; XGBoost-AFT is a censoring-aware tree model. Test MAE uses only windows resolved by day 88; C-index uses all windows.',
 'Leakage check: if labels were built from the full series, 45%, 31% and 21% of the training windows in the three folds would carry a different label than the one knowable at the cut-off.'])
ts = pd.read_csv(f'{R}/table_tau_sensitivity.csv'); ts = ts[ts.model != '[LPCF phi->E gain vs no-phi]']
ts = ts[['tau', 'model', 'n_uncens', 'MAE_end_unc', 'Cindex']]; ts.columns = ['τ', 'Model', 'Uncensored test windows', 'Endurance MAE (days)', 'C-index']
para('Threshold sensitivity (selected models):', bold=True); table(ts[ts.Model.isin(['LPCF (K=3)', 'LPCF (K=3, no phi->E)', 'K=1 (learned fixed w)', 'XGBoost', 'Random Forest', 'Train-mean'])], fs=8)
para('Endurance MAE of LPCF is not significantly better than the train-mean predictor at any threshold tested, and ranking quality (C-index) is far better for tree models. The φ→E effect: τ=0.7 Δ −1.22 days (CI [−2.73, −0.24], hurts), τ=0.8 +0.09 ([−0.15, +0.19]), τ=0.9 +0.48 ([−0.08, +0.80]).')

H1('9. Online adaptation with equal data')
ol = rd('table_online')[['model', 'n_seeds', 'MAE_log', 'MAE_log_sd', 'MAE_end_unc', 'Cindex']]; ol.columns = ['Model [update]', 'Seeds', 'MAE (log)', 'sd', 'Endurance MAE', 'C-index']
table(ol, fs=7.5)
oa = rd('table_online_adaptation_effect'); oa.columns = ['Model', 'Update', 'Gain vs static', 'lo', 'hi', 'CI excludes 0']
para('Prequential protocol: initial training on data known on day 48; every 7 days all adaptive models receive the same windows whose labels have resolved. Effect of adaptation within each model:', bold=False); table(oa, fs=8)
fig('prequential_running_mae.png', 5.4, 'Figure 5. Running prequential MAE.')

H1('10. Leave-one-game-out')
lg = rd('table_logo')[['model', 'MAE_log', 'MAE_log_sd', 'MASE', 'MAE_end_unc', 'Cindex']]; lg.columns = ['Model', 'MAE (log)', 'sd', 'MASE', 'Endurance MAE', 'C-index']
table(lg, fs=8)
lv = rd('table_logo_lpcf_vs')[['vs', 'pop_delta', 'lo', 'hi', 'end_delta', 'end_lo', 'end_hi']]; lv.columns = ['LPCF vs', 'Pop Δ', 'lo', 'hi', 'End Δ', 'lo ', 'hi ']
table(lv, fs=8)
para('For unseen games LPCF is not significantly different from K=1, uniform weights or early concatenation, is better than TFT-style selection on popularity (CI excludes zero), and is not significantly different from persistence. The comparison is limited to five games.')

H1('11. Limitations and what to do next')
bullets([
 'Google Trends must be exported (instructions are in the archive) before any claim about cross-platform fusion, or the name CPAF-Net (Cross-Platform), can be tested.',
 'Five games over one quarter is too little data to separate small effects; the rolling-origin test has 170 windows and the last fold contains a single large shock. More games with real launch-to-decay trajectories are needed, and the long monthly histories already in the archive could seed a longer-horizon dataset.',
 'No hyper-parameter search was run for any model; the gate learning rate showed that this matters. A tuned comparison with equal search budget is the next step.',
 'TFT-lite is a simplified re-implementation. DLinear is channel-independent as in the original paper.',
 'Suggested wording for the manuscript: LPCF is implemented, but on the available real data it does not outperform fixed or learned-fixed signal weighting; its latent phases are weakly aligned with lifecycle labels. The contribution should be stated as a tested hypothesis with a negative result until more data are available.'])
doc.save(f'{ROOT}/Review03_Results.docx')
