# QRT Stock Return Prediction Challenge

**54th / 1,568 participants · 52.32% accuracy · +1.01% over benchmark**

> Cross-sectional binary classification of residual stock returns. Engineered conditional mean features at industry-group granularity, built a calibrated tree + neural ensemble, and diagnosed distribution-shift leakage through adversarial validation. The core insight: in this low-SNR regime, cross-row aggregations that tree models fundamentally cannot compute on their own were the only features that reliably improved out-of-sample performance.

---

## Problem

Predict whether each stock's residual return falls above or below the cross-sectional median on a given date. The target is a per-date median split — exactly 50% of stocks are labeled 1 on each date — making this a pure ranking problem.

The dataset contains 418K training observations (156 dates × ~2,700 stocks/date) and 198K test observations across 68 unseen dates. Features are 20 lags of returns and volumes plus a four-level industry hierarchy. Dates are randomized and anonymized, so there is no temporal structure to exploit. The maximum individual feature–target correlation is 0.07, and the benchmark (Random Forest, 500 trees) scores 51.31% — signal is roughly 2% above random.

## Approach

### Feature Engineering

Tree models process each row independently — they cannot compute aggregations across stocks on the same date. But in a cross-sectional ranking problem, a stock's performance relative to its peers is more informative than its absolute return. The primary strategy was building **conditional means** (`mean(RET_i | GROUP, DATE)`) and providing these as features, giving the model genuinely new information it has no way to construct on its own.

The final feature set (48 features) has three categories:

- **Cross-row aggregations (24 features)** — conditional means at industry-group level for 5 return lags, sector/industry breadth, group momentum change, direction-agreement features, volume ratios relative to group medians, and peer-adjusted returns with thin-group fallback. These drove essentially all improvement from 51.31% to 52.32%.
- **Pre-computed summaries (10 features)** — RSI, sign fraction, cumulative returns. Features the tree could learn through sequential splitting but at a cost of 4+ splits of depth each. Pre-computing them frees splitting budget for other interactions.
- **Raw features (14 features)** — returns and volumes at lags 1–7. The signal decays past lag 7; longer lags enter only through conditional means.

**Industry Group (26 categories, ~104 stocks each) was the optimal granularity.** Sector (12 groups) was too diluted and acted as a date fingerprint. Sub-industry (175 groups, often <10 stocks) produced unreliable estimates. This was identified through correlation heterogeneity analysis in EDA and confirmed through systematic ablation.

### Model

Calibrated ensemble of three model families, chosen to maximize prediction decorrelation:

| Component | Objective | Features | Seeds | Pred. correlation with LGB |
|---|---|---|---|---|
| LightGBM | Binary cross-entropy | 48 (full) | 7 | — |
| XGBoost | MSE, labels smoothed to 0.1/0.9 | 40 (base) | 5 | 0.92 |
| MLP | MSE, labels smoothed to 0.1/0.9 | 48 (full) | 3 | 0.50 |

The tree models use different objectives because LGB learns sharper decision boundaries with binary loss while XGB extracts more signal from regression with smoothed labels — the richer gradient helps distinguish stocks near the decision boundary. The MLP (sklearn `MLPRegressor`, 32→16 hidden units, L2 regularization) provides genuinely decorrelated predictions at 0.50 correlation with the trees.

**Per-date calibration** (z-score normalization within each date) is applied before blending to align prediction scales across model families. Without calibration, the MLP's narrower prediction range gets drowned out by the trees' wider range. The final blend is 90% calibrated trees + 10% calibrated MLP.

**Asymmetric feature sets** between LGB (48 features) and XGB (40 features) add feature diversity on top of objective diversity — the models disagree not just because of different algorithms but because they see different inputs.

### Validation

- **4-fold KFold on dates** (not rows) prevents leakage through conditional means computed at the date level
- **3-repeat cross-validation** revealed that single-shuffle OOF was inflated by ~0.3% due to one anomalous fold consistently scoring 53%+
- **CV fold-averaged predictions outperformed full-data retraining** on the leaderboard (52.30% vs 51.93%) — in low-SNR, early stopping's regularization benefit outweighs the advantage of 33% more training data
- **Multi-seed averaging** (5–7 seeds per model) for variance reduction at zero bias cost

## Key Findings

**Conditional means reverse the raw signal direction.** Individual RET_1 correlates negatively with the target (stock-level reversal), but the industry-group mean of RET_1 correlates positively (group-level momentum). The model exploits both simultaneously: a stock is more likely to outperform if its group is doing well (momentum) AND its own recent return was negative (reversal — it hasn't run up yet). Conditional means provide a *different* signal, not merely an amplified version of the raw feature.

**Date fingerprinting through conditional means.** Adversarial validation (AUC 0.9995 between train and test) revealed that conditional means act as date fingerprints — with only 12 sectors, the specific combination of 12 sector-mean values uniquely identifies each date. The model can partially memorize training-date patterns through these features. Reducing from 12 conditional means across three hierarchy levels to 5 IG-only means cut the model's memorization capacity. The OOF-to-LB gap narrowed from 0.11% to 0.02%, and LB accuracy improved from 52.16% to 52.30% despite lower OOF. Rank-transforming the conditional means eliminated date-fingerprinting (adversarial AUC → 0.500) but destroyed model signal entirely — the magnitude carries information that ordinal ranking does not.

**CV predictions outperform full-data retraining.** 52.30% vs 51.93% on the leaderboard. Early stopping on a held-out fold provides an oracle for when to stop fitting noise. Full retraining removes that oracle. This is counterintuitive but directly relevant to production quantitative research — more data is not always better when the signal-to-noise ratio is this low.

**Asymmetric reversal signal.** The model predicts "winners reverse" at 53.2% accuracy but "losers bounce" at only 50.9%. Profit-taking after gains is systematic (disposition effect, rebalancing triggers), while bottom-fishing after losses is discretionary and heterogeneous. This maps to well-documented behavioral finance phenomena.

**Dropout fails in low-SNR; L2 regularization works.** PyTorch networks with 50% dropout scored 50.2–51.1%. Sklearn MLP with L2 regularization only scored 51.4%. Dropout randomly disables neurons — in a regime where every neuron carries weak signal, this destroys the signal entirely. L2 preserves all neurons but shrinks their weights, maintaining weak-but-real patterns.

**Per-date calibration is essential for multi-family ensembles.** Raw blending of tree + neural predictions hurts because of different output scales. Per-date z-score normalization before blending enabled the MLP to contribute despite only 51.4% individual accuracy. Calibrated 90/10 tree/MLP blend: 52.32% LB. Uncalibrated: 52.22% LB.

**Objective diversity matters more than model diversity.** Binary LGB + regression XGB (different objectives, same model family) added +0.12% OOF. Adding XGB with the same binary objective added nothing. The decorrelation comes from different loss landscapes emphasizing different stocks near the decision boundary, not from algorithmic differences between LGB and XGB.

## Results

| Stage | Features | OOF | LB | OOF–LB Gap |
|---|---|---|---|---|
| Benchmark (RF 500 trees) | 40 | — | 51.31% | — |
| + Conditional means | 24 | 51.55% | 51.40% | 0.15% |
| + Full feature set, META ensemble | 55 | 52.27% | 52.16% | 0.11% |
| + IG-only conditional means | 48 | 52.32% | 52.30% | 0.02% |
| **+ Calibrated MLP blend** | **48** | **52.36%** | **52.32%** | **0.04%** |

Final ranking: **54th / 1,568 participants (top 3.4%)**

## Repo Structure

```
├── README.md              — This file
├── pipeline.py            — Clean end-to-end pipeline: features → model → submission
├── eda_and_modeling.ipynb  — EDA notebook with analysis, visualizations, and full pipeline
└── experiments.md         — Research log: what was tested, what worked, what didn't, and why
```

## License

Completed as part of a private QRT quantitative research challenge. Competition data is not included per challenge rules.
