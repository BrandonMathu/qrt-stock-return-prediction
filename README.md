# QRT Stock Return Prediction Challenge

Cross-sectional binary classification of residual stock returns. Engineered conditional mean features at industry-group granularity as the primary signal source, and built a calibrated LightGBM / XGBoost / Neural Net ensemble exploiting objective diversity and prediction decorrelation across model families.

**Challenge:** [QRT Stock Return Prediction](https://challengedata.ens.fr/participants/challenges/23/)
**Username:** bmathu
**Data:** Available on the challenge page

## Results

52.32% leaderboard accuracy (benchmark: 51.31%) · **54th / 1,568 participants (top 3.4%)**

## Problem

Predict whether each stock's residual return falls above or below the cross-sectional median on a given date. The target is a per-date median split, exactly 50% of stocks are labeled 1 on each date, making this a pure ranking problem.

The dataset contains 418K training observations (156 dates × ~2,700 stocks/date) and 198K test observations across 68 unseen dates. Features are 20 lags of returns and volumes plus a four-level industry hierarchy. Dates are randomized and anonymized, eliminating any temporal structure and reducing the task to cross-sectional prediction in a very low signal-to-noise setting

## Approach

### Features

Tree models process each row independently and therefore cannot directly compute aggregations across stocks on the same date. In a cross-sectional ranking problem, relative performance within a stock's peer group is often more informative than absolute return alone. This motivated the construction of **conditional means** (`mean(RET_i | GROUP, DATE)`) and related cross-sectional features at the group level. These proved to be the main source of predictive signal: removing them reduced leaderboard accuracy from 52.32% to roughly 51.4%.

The final feature set (48 features):

- **Cross-row aggregations (24)** — conditional means at industry-group level for 5 return lags, sector/industry breadth, group momentum change, direction-agreement features, volume ratios relative to group medians, and peer-adjusted returns with thin-group fallback
- **Pre-computed summaries (10)** — RSI, sign fraction, cumulative returns. Features the tree could learn through sequential splitting but but only inefficiently
- **Raw features (14)** — returns and volumes at lags 1–7. Signal decays past lag 7 so longer lags included only through cross-sectional aggregates.

Industry Group (26 categories, ~104 stocks each) was the optimal granularity — identified through correlation heterogeneity analysis and confirmed through ablation. Sector (12 groups) was too diluted and acted as a date fingerprint. Sub-industry (175 groups, often <10 stocks) produced noisy estimates.

### Model

Calibrated ensemble of three model families, chosen to maximize prediction decorrelation:

| Component  | Objective                       | Features  | Seeds | Correlation with LGB |
| ---------- | ------------------------------- | --------- | ----- | -------------------- |
| LightGBM   | Binary cross-entropy            | 48 (full) | 7     | —                   |
| XGBoost    | MSE, labels smoothed to 0.1/0.9 | 40 (base) | 5     | 0.92                 |
| Neural net | MSE, labels smoothed to 0.1/0.9 | 48 (full) | 3     | 0.50                 |

LGB learns sharper decision boundaries with binary loss; XGB extracts more signal from regression with smoothed labels, where the richer gradient helps distinguish stocks near the decision boundary. The neural net (sklearn `MLPRegressor`, 32→16 hidden units, L2 regularization) provides genuinely decorrelated predictions — its 0.50 correlation with the trees means it disagrees on roughly half of marginal stocks.

**Per-date calibration** (z-score normalization within each date) aligns prediction scales before blending. Without it, the neural net's narrower prediction range gets drowned out by the trees' wider range. Final blend: 90% calibrated trees + 10% calibrated neural net. LGB and XGB use **asymmetric feature sets** (48 vs 40 features) for additional diversity.

### Validation

- **4-fold KFold on dates** (not rows) — prevents leakage through
  conditional means computed at the date level
- **3-repeat CV** to assess single-shuffle reliability at this
  signal level
- **CV fold-averaged predictions** for submission rather than
  full-data retraining
- **Multi-seed averaging** (5–7 seeds per model) for variance
  reduction at zero bias cost

## Key Findings

- **Conditional means reverse the raw signal direction.** Individual RET_1 correlates negatively with the target (stock-level reversal), but the group mean correlates positively (group-level momentum). The model exploits both: a stock is more likely to outperform if its group is rising and its own recent return was negative.
- **Date fingerprinting through conditional means.** Adversarial validation (AUC 0.9995) revealed that conditional means act as date identifiers — with only 12 sectors, the combination of sector-mean values uniquely fingerprints each date. Reducing to 5 IG-only means narrowed the OOF-to-LB gap from 0.11% to 0.02%.
- **Objective diversity matters more than model diversity.** Binary LGB + regression XGB added +0.12% OOF. Same-objective LGB + XGB added nothing. The gains come from different loss landscapes, not different algorithms.
- **CV predictions outperform full-data retraining.** 52.30% vs 51.93% LB — early stopping's regularization benefit outweighs 33% more training data at this signal-to-noise level.
- **Dropout fails in low-SNR; L2 works.** PyTorch with 50% dropout scored 50.2–51.1%. Sklearn MLP with L2 scored 51.4%. A likely explanation: dropout destroys weak-but-real signals that L2 preserves.
- **Asymmetric reversal signal.** Winners reverse at 53.2% accuracy, losers bounce at 50.9% — consistent with disposition effect literature.

## What Didn't Work

| Experiment                                   | Result      | Why                                                   |
| -------------------------------------------- | ----------- | ----------------------------------------------------- |
| Within-row interactions (z-scores, RET×VOL) | Neutral     | Redundant — trees learn these through splitting      |
| Categorical features (SECTOR as model input) | Overfit     | Only 156 training dates, 12 categories                |
| Per-date feature standardization             | 50.7% OOF   | Destroyed cross-date variation in conditional means   |
| Sub-industry conditional means               | Neutral     | <10 stocks per group, unreliable estimates            |
| Rank transform of conditional means          | ~51.0% OOF  | Removed magnitude information that carries signal     |
| Neural net with dropout (50%)                | 50.2–51.1% | Too aggressive for low-SNR; L2 preserves weak signals |
| Full-data retraining                         | 51.93% LB   | Loses early stopping regularization benefit           |

## Repo Structure

```
├── README.md              — This file
├── pipeline.py            — Clean end-to-end pipeline: features → model → submission
├── eda_and_modeling.ipynb  — EDA notebook with analysis, visualizations, and full pipeline
```

## License

Completed as part of a challengedata stock return prediction challenge. Competition data is not included per challenge rules.
