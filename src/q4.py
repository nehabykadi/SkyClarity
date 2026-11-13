"""
SkyClarity - Milestone 3, Q4 (prediction of next-night visibility)

Q4  Using only information that would be known the day before, how well can next-night
    sky visibility be predicted, and how does that compare with a site-only baseline?

Run from the repo root, after src/prep.py:
    python src/q4.py

Input:   data/model_ready/reports_model.csv  (2,088 cleaned observer reports, all sites)
Outputs: results/q4/
    cv_results.csv               mean absolute error (MAE) per model and held-out training year
    tuning_grid.csv              every tuning setting tried, with its cross-validated MAE
    test_metrics.csv             test MAE, share within 1 magnitude, gain over baseline with 95% CI
    error_by_group.csv           test MAE by site, wildfire season and moon phase
    permutation_importance.csv   drop in test MAE when each feature is shuffled
    predictions_test.csv         every test report with each model's prediction
    fig_q4_mae.png, fig_q4_error_by_group.png, fig_q4_importance.png
    summary.txt                  headline numbers for the write-up

Models
    Baseline     each site's median limiting magnitude in the training years (overall median
                 for sites with fewer than MIN_SITE_REPORTS training reports)
    Ridge        linear regression with an L2 penalty; numeric features standardised, site
                 one-hot encoded, missing previous-night PM2.5 filled with the training median
    Boosted      scikit-learn HistGradientBoostingRegressor; site as a categorical feature,
                 missing values handled natively

Features: Q4_FEATURES from prep.py (moon, darkness, radiance, elevation, month, previous night's
weather and smoke) plus site. That night's actual weather and smoke (Q4_EXCLUDED) are never used.
Previous-night PM2.5 is log1p-transformed, matching log_pm25_ugm3 in the cleaned data.

Tuning: leave-one-year-out cross-validation inside 2018-2022 (cv_group), scored by MAE. Each fold
is scored only on sites seen in its training years (see year_folds), as in the test set.
The test years 2023-2025 are scored once, after tuning, and never used to pick settings.

Uncertainty: several reports often share a site and night, so the 95% interval for each model's
gain over the baseline is a bootstrap that resamples whole site-nights from the test set.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep import (MOONLESS_MAX_ILLUM, MOONLESS_MAX_UP_FRAC, Q4_EXCLUDED, Q4_FEATURES,  # noqa: E402
                  TARGET, WILDFIRE_MONTHS, load_reports)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/q4"
SEED = 5502
N_BOOT = 2000
N_PERM = 30
MIN_SITE_REPORTS = 5    # fewer training reports than this: baseline uses the overall median
MIN_TEST_REPORTS = 10   # sites with fewer test reports are left out of the by-site table

NUMERIC = [("prev_log_pm25_ugm3" if c == "prev_pm25_ugm3" else c) for c in Q4_FEATURES]
FEATURES = NUMERIC + ["site_id"]
assert not set(FEATURES) & set(Q4_EXCLUDED), "Q4 must not use same-night weather or smoke"

RIDGE_GRID = {"model__alpha": np.logspace(-4, 3, 15)}
BOOSTED_GRID = {
    "loss": ["absolute_error", "squared_error"],
    "learning_rate": [0.03, 0.1, 0.2],
    "max_iter": [100, 300],
    "max_leaf_nodes": [7, 15],
    "min_samples_leaf": [10, 20, 50],
    "l2_regularization": [0.0, 1.0, 10.0],
}
MODELS = ["Baseline", "Ridge", "Boosted"]

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9893"
COLORS = {"Baseline": GREY, "Ridge": BLUE, "Boosted": ORANGE}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- data and models
def load():
    r = load_reports()
    r["prev_log_pm25_ugm3"] = np.log1p(r.prev_pm25_ugm3)
    r["site_id"] = pd.Categorical(r.site_id, categories=sorted(r.site_id.unique()))
    r["is_moonless"] = ((r.moon_up_frac <= MOONLESS_MAX_UP_FRAC)
                        | (r.moon_illum_frac <= MOONLESS_MAX_ILLUM)).astype(int)
    return r[r.split == "train"].reset_index(drop=True), r[r.split == "test"].reset_index(drop=True)


class SiteMedian:
    """Baseline: training median per site, overall median for sparsely reported sites."""

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        g = y.groupby(X.site_id, observed=True)
        self.overall_ = y.median()
        self.medians_ = g.median()[g.size() >= MIN_SITE_REPORTS]
        return self

    def predict(self, X):
        return X.site_id.astype(str).map(self.medians_).fillna(self.overall_).to_numpy(float)


def ridge():
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("site", OneHotEncoder(handle_unknown="ignore"), ["site_id"]),
    ])
    return Pipeline([("prep", prep), ("model", Ridge())])


def boosted():
    return HistGradientBoostingRegressor(categorical_features="from_dtype", early_stopping=False,
                                         random_state=SEED)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def year_folds(train):
    """Leave-one-year-out folds, each scored only on sites that appear in its training years.

    Every test-set site has training reports, so a fold should not be scored on a site its
    models have never seen. Without this, the 2022 fold is mostly Crestone (107 of 205 reports),
    whose first reports are in 2022, and it measures new-site rather than new-year error.
    """
    folds = []
    for tr_idx, va_idx in LeaveOneGroupOut().split(train, groups=train.cv_group):
        seen = train.site_id.iloc[tr_idx].unique()
        folds.append((tr_idx, va_idx[train.site_id.iloc[va_idx].isin(seen).to_numpy()]))
    return folds


# ---------------------------------------------------------------- tuning (training years only)
def tune(train):
    X, y, groups = train[FEATURES], train[TARGET], train.cv_group
    folds = year_folds(train)
    searches, grid_rows, fold_rows = {}, [], []
    for name, est, grid in [("Ridge", ridge(), RIDGE_GRID), ("Boosted", boosted(), BOOSTED_GRID)]:
        s = GridSearchCV(est, grid, scoring="neg_mean_absolute_error", cv=folds, n_jobs=-1)
        s.fit(X, y)
        searches[name] = s
        cv = pd.DataFrame(s.cv_results_)
        for _, row in cv.iterrows():
            grid_rows.append({"model": name, "params": row.params,
                              "cv_mae": -row.mean_test_score, "cv_mae_sd": row.std_test_score,
                              "rank": row.rank_test_score})
        best = cv.loc[s.best_index_]
        for k, (_, va_idx) in enumerate(folds):
            fold_rows.append({"model": name, "held_out_year": groups.iloc[va_idx[0]],
                              "n_scored": len(va_idx), "mae": -best[f"split{k}_test_score"]})

    for tr_idx, va_idx in folds:   # baseline scored on the same folds
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        fold_rows.append({"model": "Baseline", "held_out_year": groups.iloc[va_idx[0]],
                          "n_scored": len(va_idx),
                          "mae": mae(va[TARGET], SiteMedian().fit(tr[FEATURES], tr[TARGET])
                                     .predict(va[FEATURES]))})
    return searches, pd.DataFrame(grid_rows), pd.DataFrame(fold_rows)


# ---------------------------------------------------------------- test evaluation (once)
def fit_final(train, searches):
    fitted = {"Baseline": SiteMedian().fit(train[FEATURES], train[TARGET])}
    for name, s in searches.items():
        fitted[name] = s.best_estimator_   # GridSearchCV refits the best setting on all of 2018-2022
    return fitted


def gain_ci(test, rng):
    """95% interval of (baseline MAE - model MAE), resampling whole site-nights."""
    cluster = test.groupby(["site_id", "night_date"], observed=True).ngroup().to_numpy()
    k = cluster.max() + 1
    err = {m: np.abs(test[TARGET] - test[f"pred_{m}"]).to_numpy() for m in MODELS}
    sums = {m: np.bincount(cluster, err[m], k) for m in MODELS}
    n = np.bincount(cluster, minlength=k)
    out = {}
    for m in ["Ridge", "Boosted"]:
        diffs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, k, k)
            w = np.bincount(idx, minlength=k)
            diffs.append((w @ sums["Baseline"] - w @ sums[m]) / (w @ n))
        out[m] = np.percentile(diffs, [2.5, 97.5])
    return out


def test_metrics(test, rng):
    ci = gain_ci(test, rng)
    base = mae(test[TARGET], test.pred_Baseline)
    rows = []
    for m in MODELS:
        err = np.abs(test[TARGET] - test[f"pred_{m}"])
        lo, hi = ci.get(m, (np.nan, np.nan))
        rows.append({"model": m, "test_mae": err.mean(), "within_1_mag": (err <= 1).mean(),
                     "gain_vs_baseline": base - err.mean(), "gain_ci_low": lo, "gain_ci_high": hi})
    return pd.DataFrame(rows)


def error_by_group(test):
    t = test.assign(season=np.where(test.month.isin(WILDFIRE_MONTHS), "Jun-Sep", "Oct-May"),
                    moon=np.where(test.is_moonless == 1, "moonless", "moonlit"),
                    site=test.site_id.astype(str))
    for m in MODELS:
        t[m] = np.abs(t[TARGET] - t[f"pred_{m}"])
    rows = []
    for by in ["site", "season", "moon"]:
        g = t.groupby(by)
        tab = g[MODELS].mean().assign(n=g.size())
        if by == "site":
            tab = tab[tab.n >= MIN_TEST_REPORTS].sort_values("n", ascending=False)
        rows.append(tab.reset_index().rename(columns={by: "group"}).assign(grouping=by))
    return pd.concat(rows)[["grouping", "group", "n"] + MODELS]


def importance(test, fitted):
    rows = []
    for m in ["Ridge", "Boosted"]:
        pi = permutation_importance(fitted[m], test[FEATURES], test[TARGET], n_repeats=N_PERM,
                                    scoring="neg_mean_absolute_error", random_state=SEED)
        rows.append(pd.DataFrame({"model": m, "feature": FEATURES,
                                  "mae_increase": pi.importances_mean, "sd": pi.importances_std}))
    return pd.concat(rows).sort_values(["model", "mae_increase"], ascending=[True, False])


# ---------------------------------------------------------------- figures
def fig_mae(folds, metrics):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2]})
    piv = folds.pivot(index="held_out_year", columns="model", values="mae")[MODELS]
    x, w = np.arange(len(piv)), 0.26
    for i, m in enumerate(MODELS):
        a.bar(x + (i - 1) * w, piv[m], w, color=COLORS[m], label=m)
    a.set_xticks(x, piv.index)
    a.set(title="Cross-validation, by held-out training year", ylabel="MAE (magnitudes)",
          ylim=(0, piv.max().max() * 1.25))
    a.legend(frameon=False, ncols=3, loc="upper left")

    met = metrics.set_index("model").loc[MODELS]
    b.bar(MODELS, met.test_mae, color=[COLORS[m] for m in MODELS], width=0.6)
    for i, v in enumerate(met.test_mae):
        b.text(i, v + 0.02, f"{v:.2f}", ha="center", color=INK2)
    b.set(title="Test years 2023-2025", ylim=(0, max(met.test_mae) * 1.2))
    for ax in (a, b):
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_q4_mae.png", dpi=150)
    plt.close(fig)


def fig_groups(groups):
    parts = [("site", "By site (10+ test reports)"), ("season", "By season"), ("moon", "By moon")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [len(groups[groups.grouping == g]) for g, _ in parts]})
    for ax, (g, title) in zip(axes, parts):
        t = groups[groups.grouping == g]
        x, w = np.arange(len(t)), 0.26
        for i, m in enumerate(MODELS):
            ax.bar(x + (i - 1) * w, t[m], w, color=COLORS[m], label=m)
        ax.set_xticks(x, [f"{s}\n(n={n})" for s, n in zip(t.group, t.n)],
                      rotation=45 if g == "site" else 0, ha="right" if g == "site" else "center",
                      fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test MAE (magnitudes)")
    axes[0].legend(frameon=False, ncols=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_q4_error_by_group.png", dpi=150)
    plt.close(fig)


def fig_importance(imp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    for ax, m in zip(axes, ["Ridge", "Boosted"]):
        t = imp[imp.model == m].sort_values("mae_increase")
        ax.barh(t.feature, t.mae_increase, xerr=t.sd, color=COLORS[m], ecolor=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(m)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
    fig.supxlabel("Increase in test MAE when the feature is shuffled", color=INK2)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_q4_importance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    train, test = load()

    searches, grid, folds = tune(train)
    grid.sort_values(["model", "rank"]).to_csv(RESULTS / "tuning_grid.csv", index=False,
                                               float_format="%.4f")
    folds.to_csv(RESULTS / "cv_results.csv", index=False, float_format="%.4f")

    fitted = fit_final(train, searches)
    for m in MODELS:
        test[f"pred_{m}"] = fitted[m].predict(test[FEATURES])
    metrics = test_metrics(test, rng)
    groups = error_by_group(test)
    imp = importance(test, fitted)

    metrics.to_csv(RESULTS / "test_metrics.csv", index=False, float_format="%.4f")
    groups.to_csv(RESULTS / "error_by_group.csv", index=False, float_format="%.4f")
    imp.to_csv(RESULTS / "permutation_importance.csv", index=False, float_format="%.4f")
    test[["gan_id", "night_date", "site_id", TARGET] + [f"pred_{m}" for m in MODELS]].to_csv(
        RESULTS / "predictions_test.csv", index=False, float_format="%.3f", date_format="%Y-%m-%d")
    fig_mae(folds, metrics)
    fig_groups(groups)
    fig_importance(imp)

    cv = folds.groupby("model").mae.agg(["mean", "std"]).loc[MODELS]
    met = metrics.set_index("model")
    top = lambda m: ", ".join(imp[imp.model == m].head(4).feature)
    lines = [
        "SkyClarity - Q4 results", f"seed {SEED}, {N_BOOT} site-night bootstrap resamples, "
        f"{N_PERM} permutation repeats", "",
        f"Train 2018-2022: {len(train):,} reports; test 2023-2025: {len(test):,} reports",
        f"Features ({len(FEATURES)}): " + ", ".join(FEATURES), "",
        "Chosen settings (leave-one-year-out CV on the training years)",
        f"  Ridge:   alpha = {searches['Ridge'].best_params_['model__alpha']:.3g}",
        "  Boosted: " + ", ".join(f"{k} = {v}" for k, v in searches["Boosted"].best_params_.items()),
        "",
        "Cross-validated MAE (mean and sd over the 5 held-out years)",
        "  " + cv.map("{:.3f}".format).to_string().replace("\n", "\n  "),
        "",
        "Test MAE, 2023-2025 (scored once)",
    ] + [
        f"  {m:<8} MAE {met.loc[m, 'test_mae']:.3f}, within 1 magnitude "
        f"{met.loc[m, 'within_1_mag']:.1%}"
        + ("" if m == "Baseline" else
           f", gain over baseline {met.loc[m, 'gain_vs_baseline']:+.3f} "
           f"(95% CI {met.loc[m, 'gain_ci_low']:+.3f} to {met.loc[m, 'gain_ci_high']:+.3f})")
        for m in MODELS
    ] + [
        "",
        "Test MAE by group",
        "  " + groups.to_string(index=False, float_format="{:.2f}".format).replace("\n", "\n  "),
        "",
        f"Most important features (permutation, test set): Ridge: {top('Ridge')}; "
        f"Boosted: {top('Boosted')}",
        "",
        "Caveats: 'within 1 magnitude' favours the baseline, whose whole-number predictions match the",
        "whole-number target exactly or by 1, while the models predict in between; compare MAE first.",
        "'Day-before information' is the previous night's actual conditions, not a forecast;",
        "Pawnee Buttes and RMNP have no test reports and several sites have fewer than 10;",
        f"only {int((test.hms_smoke_level > 0).sum())} test reports fall on smoke days.",
    ]
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
