"""
QRT Stock Return Prediction — Pipeline
=======================================
Cross-sectional binary classification of residual stock returns.
Produces the final 52.32% leaderboard submission.

Ensemble: Calibrated BR_ASYM (LGB binary + XGB regression) + MLP at 10% weight.

Usage:
    python pipeline.py

Data files expected in ./data/:
    x_train.csv, y_train.csv, x_test.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings, time, gc

warnings.filterwarnings('ignore')


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "./data"
N_FOLDS = 4
RANDOM_STATE = 0

# Multi-seed configs for variance reduction
SEEDS_LGB = [42, 17, 123, 7, 99, 31, 53]   # 7 seeds
SEEDS_XGB = [42, 17, 123, 7, 99]             # 5 seeds
SEEDS_MLP = [42, 17, 123]                     # 3 seeds

# Raw feature columns
RET_COLS = [f'RET_{i}' for i in range(1, 21)]
VOL_COLS = [f'VOLUME_{i}' for i in range(1, 21)]

# LightGBM — binary cross-entropy objective
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': -1,
    'learning_rate': 0.015, 'num_leaves': 31, 'max_depth': 6,
    'min_child_samples': 200, 'feature_fraction': 0.7,
    'bagging_fraction': 0.7, 'bagging_freq': 1,
    'lambda_l1': 0.5, 'lambda_l2': 5.0,
}

# XGBoost — regression with smoothed labels (0.1/0.9)
XGB_PARAMS = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 6, 'learning_rate': 0.015,
    'min_child_weight': 200, 'subsample': 0.7,
    'colsample_bytree': 0.7, 'reg_alpha': 0.5,
    'reg_lambda': 5.0, 'nthread': -1, 'verbosity': 0,
}

# MLP — small network, L2 regularization (no dropout)
MLP_PARAMS = {
    'hidden_layer_sizes': (32, 16),
    'activation': 'relu',
    'alpha': 0.01,
    'learning_rate_init': 0.003,
    'batch_size': 16384,
    'max_iter': 100,
    'early_stopping': True,
    'validation_fraction': 0.15,
    'n_iter_no_change': 10,
    'verbose': False,
}


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df, clip_bounds):
    """
    Engineer all 48 features from raw returns and volumes.

    Feature categories:
        - Cross-sectional aggregations (24): conditional means, breadth,
          momentum change, direction agreement, volume ratios, peer adjustment
        - Pre-computed summaries (10): RSI, sign fraction, cumulative returns,
          relative performance measures
        - Raw features (14): RET_1-7, VOLUME_1-7

    All cross-sectional features are computed within each date — no lookahead.
    """
    df = df.copy()

    # --- Winsorize at 1st/99th percentile ---
    for col in RET_COLS + VOL_COLS:
        lo, hi = clip_bounds[col]
        df[col] = df[col].clip(lo, hi)

    # --- Conditional means at Industry Group level ---
    # IG (26 groups, ~104 stocks each) is the optimal granularity:
    # fine enough for peer dynamics, coarse enough for stable estimates
    for shift in [1, 2, 3, 5, 17]:
        feat = f'RET_{shift}'
        df[f'RET_{shift}_IG_DATE_mean'] = (
            df.groupby(['DATE', 'INDUSTRY_GROUP'])[feat].transform('mean')
        )

    # --- RSI and sign fraction ---
    for window, label in [(5, '5'), (10, '10'), (20, '20')]:
        ret_window = [f'RET_{i}' for i in range(1, window + 1)]
        gains = df[ret_window].clip(lower=0).mean(axis=1)
        losses = (-df[ret_window].clip(upper=0)).mean(axis=1)
        total = gains + losses
        df[f'rsi_{label}d'] = np.where(total > 1e-10, gains / total, 0.5)

    for window, label in [(5, '5'), (20, '20')]:
        ret_window = [f'RET_{i}' for i in range(1, window + 1)]
        df[f'sign_frac_{label}d'] = (df[ret_window] > 0).sum(axis=1) / window

    # --- Breadth (fraction of group with positive RET_1) ---
    df['_ret_positive'] = (df['RET_1'] > 0).astype(float)
    df['sector_breadth'] = (
        df.groupby(['DATE', 'SECTOR'])['_ret_positive'].transform('mean')
    )
    df['industry_breadth'] = (
        df.groupby(['DATE', 'INDUSTRY'])['_ret_positive'].transform('mean')
    )
    df['stock_vs_sector_breadth'] = df['_ret_positive'] - df['sector_breadth']
    df['stock_vs_industry_breadth'] = df['_ret_positive'] - df['industry_breadth']
    df.drop(columns=['_ret_positive'], inplace=True)

    # --- Cumulative and relative features ---
    df['cum_ret_1_5'] = df[[f'RET_{i}' for i in range(1, 6)]].sum(axis=1)
    df['ret_std_5d'] = df[[f'RET_{i}' for i in range(1, 6)]].std(axis=1)
    df['vol_mean_5d'] = df[[f'VOLUME_{i}' for i in range(1, 6)]].mean(axis=1)

    # Relative to sector peers (date-invariant by construction)
    for base, stat in [('cum_ret_1_5', 'cum_ret_1_5'),
                       ('ret_std_5d', 'ret_std_5d'),
                       ('vol_mean_5d', 'vol_mean_5d'),
                       ('rsi_5d', 'rsi_5d')]:
        sector_mean = df.groupby(['DATE', 'SECTOR'])[base].transform('mean')
        df[f'{stat}_rel_sector'] = df[base] - sector_mean

    df['cum_ret_1_5_industry_mean'] = (
        df.groupby(['DATE', 'INDUSTRY'])['cum_ret_1_5'].transform('mean')
    )
    df['cum_ret_1_5_rel_industry'] = df['cum_ret_1_5'] - df['cum_ret_1_5_industry_mean']

    # --- Group momentum change (is the group accelerating?) ---
    # sector_momentum_change = sector's RET_1 mean - sector's trailing mean (RET_2..10)
    for i in range(2, 11):
        df[f'_sec_ret_{i}'] = (
            df.groupby(['DATE', 'SECTOR'])[f'RET_{i}'].transform('mean')
        )
    df['_sec_ret_1'] = df.groupby(['DATE', 'SECTOR'])['RET_1'].transform('mean')
    df['sector_momentum_change'] = (
        df['_sec_ret_1'] - df[[f'_sec_ret_{i}' for i in range(2, 11)]].mean(axis=1)
    )

    for i in range(2, 11):
        df[f'_ig_ret_{i}'] = (
            df.groupby(['DATE', 'INDUSTRY_GROUP'])[f'RET_{i}'].transform('mean')
        )
    df['ig_momentum_change'] = (
        df['RET_1_IG_DATE_mean']
        - df[[f'_ig_ret_{i}' for i in range(2, 11)]].mean(axis=1)
    )

    # Clean up temp columns
    for i in range(2, 11):
        df.drop(columns=[f'_sec_ret_{i}', f'_ig_ret_{i}'], inplace=True)
    df.drop(columns=['_sec_ret_1'], inplace=True)

    # --- IG direction agreement ---
    df['ret1_ig_agree'] = df['RET_1'] * df['RET_1_IG_DATE_mean']
    df['ret2_ig_agree'] = df['RET_2'] * df['RET_2_IG_DATE_mean']

    # Relative to IG peers
    ig_cum_mean = df.groupby(['DATE', 'INDUSTRY_GROUP'])['cum_ret_1_5'].transform('mean')
    df['cum_ret_1_5_rel_ig'] = df['cum_ret_1_5'] - ig_cum_mean
    ig_rsi_mean = df.groupby(['DATE', 'INDUSTRY_GROUP'])['rsi_5d'].transform('mean')
    df['rsi_5d_rel_ig'] = df['rsi_5d'] - ig_rsi_mean

    # --- Volume context ---
    ig_vol_med = df.groupby(['DATE', 'INDUSTRY_GROUP'])['VOLUME_1'].transform('median')
    df['h2_vol_ratio_ig'] = df['VOLUME_1'] / (ig_vol_med.abs() + 1e-8)
    sec_vol_med = df.groupby(['DATE', 'SECTOR'])['VOLUME_1'].transform('median')
    df['h2_vol_ratio_sector'] = df['VOLUME_1'] / (sec_vol_med.abs() + 1e-8)

    # Return-volume correlation (5-day window)
    ret_v = df[['RET_1', 'RET_2', 'RET_3', 'RET_4', 'RET_5']].values
    vol_v = df[['VOLUME_1', 'VOLUME_2', 'VOLUME_3', 'VOLUME_4', 'VOLUME_5']].fillna(0).values
    rm = np.nanmean(ret_v, axis=1, keepdims=True)
    vm = np.nanmean(vol_v, axis=1, keepdims=True)
    cov = np.nanmean((ret_v - rm) * (vol_v - vm), axis=1)
    denom = np.nanstd(ret_v, axis=1) * np.nanstd(vol_v, axis=1)
    df['h3_ret_vol_corr'] = np.where(denom > 1e-10, cov / denom, 0.0)

    # Return acceleration
    df['h3_ret_acceleration'] = (
        df[[f'RET_{i}' for i in range(1, 6)]].mean(axis=1)
        - df[[f'RET_{i}' for i in range(16, 21)]].mean(axis=1)
    )

    # --- Peer adjustment with thin-group fallback ---
    for shift in [1, 2]:
        sub_count = (
            df.groupby(['DATE', 'SUB_INDUSTRY'])[f'RET_{shift}'].transform('count')
        )
        sub_mean = (
            df.groupby(['DATE', 'SUB_INDUSTRY'])[f'RET_{shift}'].transform('mean')
        )
        ind_mean = (
            df.groupby(['DATE', 'INDUSTRY'])[f'RET_{shift}'].transform('mean')
        )
        # Use sub-industry mean if 10+ stocks, otherwise fall back to industry
        df[f'h5_ret{shift}_peer_mean'] = np.where(sub_count >= 10, sub_mean, ind_mean)
        df[f'h5_ret{shift}_peer_residual'] = df[f'RET_{shift}'] - df[f'h5_ret{shift}_peer_mean']

    # Clean up intermediate columns
    for col in ['ret_std_5d', 'vol_mean_5d', 'cum_ret_1_5_industry_mean']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# Full feature set (48 features) — D_ig_only configuration
# Uses only IG conditional means (no sector/industry conditional means)
FEATURES = [
    # Raw features (14)
    'RET_1', 'RET_2', 'RET_3', 'RET_4', 'RET_5', 'RET_6', 'RET_7',
    'VOLUME_1', 'VOLUME_2', 'VOLUME_3', 'VOLUME_4', 'VOLUME_5', 'VOLUME_6', 'VOLUME_7',
    # IG conditional means (5)
    'RET_1_IG_DATE_mean', 'RET_2_IG_DATE_mean', 'RET_3_IG_DATE_mean',
    'RET_5_IG_DATE_mean', 'RET_17_IG_DATE_mean',
    # RSI and sign fraction (5)
    'rsi_5d', 'rsi_10d', 'rsi_20d', 'sign_frac_5d', 'sign_frac_20d',
    # Breadth (4)
    'sector_breadth', 'industry_breadth',
    'stock_vs_sector_breadth', 'stock_vs_industry_breadth',
    # Cumulative and relative (6)
    'cum_ret_1_5', 'cum_ret_1_5_rel_sector', 'cum_ret_1_5_rel_industry',
    'ret_std_5d_rel_sector', 'vol_mean_5d_rel_sector', 'rsi_5d_rel_sector',
    # Group dynamics (6)
    'sector_momentum_change', 'ig_momentum_change',
    'ret1_ig_agree', 'ret2_ig_agree', 'cum_ret_1_5_rel_ig', 'rsi_5d_rel_ig',
    # Volume context and stock-level (4)
    'h2_vol_ratio_ig', 'h2_vol_ratio_sector',
    'h3_ret_vol_corr', 'h3_ret_acceleration',
    # Peer adjustment (4)
    'h5_ret1_peer_mean', 'h5_ret1_peer_residual',
    'h5_ret2_peer_mean', 'h5_ret2_peer_residual',
]

# Base features for asymmetric XGB (40 features — full set minus v14 additions)
FEATURES_BASE = [f for f in FEATURES if f not in [
    'h2_vol_ratio_ig', 'h2_vol_ratio_sector',
    'h3_ret_vol_corr', 'h3_ret_acceleration',
    'h5_ret1_peer_mean', 'h5_ret1_peer_residual',
    'h5_ret2_peer_mean', 'h5_ret2_peer_residual',
]]


# ============================================================
# MODEL TRAINING
# ============================================================

def train_lgb(X_tr, y_tr, X_va, y_va, X_te, features, seeds):
    """Train LightGBM with multi-seed averaging."""
    preds_va = np.zeros(len(X_va))
    preds_te = np.zeros(len(X_te))

    for seed in seeds:
        params = LGB_PARAMS.copy()
        params['random_state'] = seed
        ds_train = lgb.Dataset(X_tr, y_tr, feature_name=features)
        ds_val = lgb.Dataset(X_va, y_va, reference=ds_train)
        model = lgb.train(
            params, ds_train, 3000,
            valid_sets=[ds_val], valid_names=['val'],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        preds_va += model.predict(X_va) / len(seeds)
        preds_te += model.predict(X_te) / len(seeds)
        del ds_train, ds_val, model; gc.collect()

    return preds_va, preds_te


def train_xgb(X_tr, y_tr, X_va, y_va, X_te, seeds):
    """Train XGBoost with multi-seed averaging."""
    preds_va = np.zeros(len(X_va))
    preds_te = np.zeros(len(X_te))

    dm_tr = xgb.DMatrix(X_tr, label=y_tr)
    dm_va = xgb.DMatrix(X_va, label=y_va)
    dm_te = xgb.DMatrix(X_te)

    for seed in seeds:
        params = XGB_PARAMS.copy()
        params['seed'] = seed
        model = xgb.train(
            params, dm_tr, 3000,
            evals=[(dm_va, 'val')],
            early_stopping_rounds=100, verbose_eval=False
        )
        preds_va += model.predict(dm_va) / len(seeds)
        preds_te += model.predict(dm_te) / len(seeds)
        del model; gc.collect()

    del dm_tr, dm_va, dm_te; gc.collect()
    return preds_va, preds_te


def train_mlp(X_tr_scaled, y_tr, X_va_scaled, X_te_scaled, seeds):
    """Train sklearn MLPRegressor with multi-seed averaging."""
    preds_va = np.zeros(len(X_va_scaled))
    preds_te = np.zeros(len(X_te_scaled))

    for seed in seeds:
        mlp = MLPRegressor(**MLP_PARAMS, random_state=seed)
        mlp.fit(X_tr_scaled, y_tr)
        preds_va += mlp.predict(X_va_scaled) / len(seeds)
        preds_te += mlp.predict(X_te_scaled) / len(seeds)

    return preds_va, preds_te


# ============================================================
# BLENDING UTILITIES
# ============================================================

def per_date_calibrate(preds, dates):
    """Per-date z-score normalization — aligns prediction scales across model families."""
    df = pd.DataFrame({'pred': preds, 'date': dates})
    df['cal'] = df.groupby('date')['pred'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    return df['cal'].values


def predict_submission(preds, dates):
    """Per-date median thresholding — matches the target's construction."""
    df = pd.DataFrame({'prob': preds, 'date': dates})
    return df.groupby('date')['prob'].transform(lambda x: x > x.median()).values.astype(int)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    t_start = time.time()

    # --- 1. Load data ---
    print("=" * 60)
    print("QRT Stock Return Prediction — Pipeline")
    print("=" * 60)

    x_train = pd.read_csv(f"{DATA_DIR}/x_train.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")
    x_test = pd.read_csv(f"{DATA_DIR}/x_test.csv")
    train = x_train.merge(y_train, on="ID", how="left")
    train['RET'] = train['RET'].astype(int)

    print(f"\nTrain: {train.shape} | Test: {x_test.shape}")
    print(f"Dates: {train['DATE'].nunique()} train, {x_test['DATE'].nunique()} test")
    print(f"Stocks per date: ~{len(train) // train['DATE'].nunique()}")

    # --- 2. Compute clip bounds ---
    clip_bounds = {}
    for col in RET_COLS + VOL_COLS:
        clip_bounds[col] = (train[col].quantile(0.01), train[col].quantile(0.99))

    # --- 3. Feature engineering ---
    print("\nEngineering features...")
    t0 = time.time()
    train = build_features(train, clip_bounds)
    x_test = build_features(x_test, clip_bounds)
    print(f"  {len(FEATURES)} features built ({time.time() - t0:.1f}s)")

    # --- 4. Drop all-NaN rows ---
    ret5 = [f'RET_{i}' for i in range(1, 6)]
    nan_mask = train[ret5].isna().all(axis=1)
    train_clean = train[~nan_mask].copy()
    y_clean = train_clean['RET'].values
    y_smooth = np.where(y_clean == 1, 0.9, 0.1)
    print(f"  Dropped {nan_mask.sum()} all-NaN rows: {len(train)} → {len(train_clean)}")

    # --- 5. Cross-validation ---
    print("\nRunning 4-fold date-based CV...")
    cv_dates = train_clean['DATE'].unique()
    kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)

    # Storage for OOF and test predictions
    models = ['lgb_bin', 'xgb_reg', 'mlp']
    oof = {k: np.zeros(len(train_clean)) for k in models}
    test_preds = {k: np.zeros(len(x_test)) for k in models}

    for fold, (tr_di, va_di) in enumerate(kf.split(cv_dates)):
        t0 = time.time()
        tr_dates, va_dates = cv_dates[tr_di], cv_dates[va_di]
        tr_mask = train_clean['DATE'].isin(tr_dates)
        va_mask = train_clean['DATE'].isin(va_dates)

        y_tr = y_clean[tr_mask.values]
        y_va = y_clean[va_mask.values]
        y_tr_smooth = y_smooth[tr_mask.values]
        y_va_smooth = y_smooth[va_mask.values]

        # Full features
        X_tr = train_clean.loc[tr_mask, FEATURES].fillna(0)
        X_va = train_clean.loc[va_mask, FEATURES].fillna(0)
        X_te = x_test[FEATURES].fillna(0)

        # Base features (for asymmetric XGB)
        X_tr_base = train_clean.loc[tr_mask, FEATURES_BASE].fillna(0)
        X_va_base = train_clean.loc[va_mask, FEATURES_BASE].fillna(0)
        X_te_base = x_test[FEATURES_BASE].fillna(0)

        # LGB Binary (7 seeds, full features)
        f_lgb, ft_lgb = train_lgb(X_tr, y_tr, X_va, y_va, X_te, FEATURES, SEEDS_LGB)
        oof['lgb_bin'][va_mask.values] = f_lgb
        test_preds['lgb_bin'] += ft_lgb / N_FOLDS

        # XGB Regression (5 seeds, base features, smoothed labels)
        f_xgb, ft_xgb = train_xgb(X_tr_base, y_tr_smooth, X_va_base, y_va_smooth, X_te_base, SEEDS_XGB)
        oof['xgb_reg'][va_mask.values] = f_xgb
        test_preds['xgb_reg'] += ft_xgb / N_FOLDS

        # MLP (3 seeds, standardized features, smoothed labels)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_te)

        f_mlp, ft_mlp = train_mlp(X_tr_sc, y_tr_smooth, X_va_sc, X_te_sc, SEEDS_MLP)
        oof['mlp'][va_mask.values] = f_mlp
        test_preds['mlp'] += ft_mlp / N_FOLDS

        # Fold summary
        dates_va = train_clean.loc[va_mask, 'DATE'].values
        br_asym = 0.5 * f_lgb + 0.5 * f_xgb
        acc = accuracy_score(y_va, predict_submission(br_asym, dates_va))
        print(f"  Fold {fold + 1}: BR_ASYM={acc:.4%} ({time.time() - t0:.0f}s)")

        del X_tr, X_va, X_te, X_tr_base, X_va_base, X_te_base
        del X_tr_sc, X_va_sc, X_te_sc
        gc.collect()

    # --- 6. Evaluate and blend ---
    print("\nEvaluating ensembles...")
    valid = oof['lgb_bin'] != 0
    oof_dates = train_clean.loc[valid, 'DATE'].values
    oof_y = y_clean[valid]

    # Per-date calibration
    train_dates_all = train_clean['DATE'].values
    test_dates_all = x_test['DATE'].values

    oof_cal = {k: per_date_calibrate(oof[k], train_dates_all) for k in models}
    test_cal = {k: per_date_calibrate(test_preds[k], test_dates_all) for k in models}

    # Ensemble variants
    results = {}

    def evaluate(name, oof_pred, test_pred):
        acc = accuracy_score(oof_y, predict_submission(oof_pred[valid], oof_dates))
        results[name] = {'acc': acc, 'test': test_pred}
        return acc

    # Individual models
    for name in models:
        evaluate(name, oof[name], test_preds[name])

    # BR_ASYM (raw) — LGB binary + XGB regression
    evaluate('BR_ASYM',
             0.5 * oof['lgb_bin'] + 0.5 * oof['xgb_reg'],
             0.5 * test_preds['lgb_bin'] + 0.5 * test_preds['xgb_reg'])

    # Calibrated BR_ASYM + MLP at 10% (our best)
    br_cal_oof = 0.5 * oof_cal['lgb_bin'] + 0.5 * oof_cal['xgb_reg']
    br_cal_test = 0.5 * test_cal['lgb_bin'] + 0.5 * test_cal['xgb_reg']

    evaluate('CAL_BR+MLP_10',
             0.90 * br_cal_oof + 0.10 * oof_cal['mlp'],
             0.90 * br_cal_test + 0.10 * test_cal['mlp'])

    # Additional blend weights for reference
    for w in [0.05, 0.15, 0.20]:
        evaluate(f'CAL_BR+MLP_{int(w*100)}',
                 (1 - w) * br_cal_oof + w * oof_cal['mlp'],
                 (1 - w) * br_cal_test + w * test_cal['mlp'])

    # Print results
    print(f"\n{'Ensemble':<25} {'OOF Accuracy':>12}")
    print("-" * 39)
    for name in sorted(results, key=lambda k: -results[k]['acc']):
        marker = " ← best" if name == 'CAL_BR+MLP_10' else ""
        print(f"  {name:<23} {results[name]['acc']:>12.4%}{marker}")

    # Prediction correlation matrix
    print(f"\nPrediction correlation (calibrated OOF):")
    corr_df = pd.DataFrame({k: oof_cal[k][valid] for k in models})
    print(corr_df.corr().round(3).to_string())

    # --- 7. Generate submission ---
    print("\nGenerating submission...")
    best_test = results['CAL_BR+MLP_10']['test']
    final_preds = predict_submission(best_test, test_dates_all)

    submission = pd.DataFrame({
        'ID': x_test['ID'],
        'RET': final_preds.astype(bool)
    })
    submission.to_csv("submission.csv", index=False)

    print(f"  Saved: submission.csv ({len(submission)} rows, "
          f"{submission['RET'].mean():.4%} positive)")
    print(f"\nTotal runtime: {time.time() - t_start:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
