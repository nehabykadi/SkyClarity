import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import Q3_FEATURES, TARGET, load_reports

# ---------------- CONFIG (edit if your column names differ) ----------------
SITE_COL = "site_id"
DATE_COL = "night_date"                # night identifier, used for the bootstrap
SMOKE_SEASON_MONTHS = [6, 7, 8, 9]  # wildfire season, Jun-Sep
N_BOOT = 2000                       # OLS bootstrap reps
N_BOOT_ORD = 300                    # ordinal bootstrap reps (slower)
MIN_PER_GROUP = 5                   # min reports in each season group per site
SEED = 42
OUT = "results/q2b_q3"
# ---------------------------------------------------------------------------

os.makedirs(OUT, exist_ok=True)
print("Running Q2 Part B and Q3 (ordinal bootstrap is the slow step, about a few minutes)...", flush=True)
rng = np.random.default_rng(SEED)

df = load_reports()
for c in (SITE_COL, DATE_COL, TARGET):
    if c not in df.columns:
        raise KeyError(f"Column '{c}' not in reports. Available: {list(df.columns)}")
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.dropna(subset=[TARGET]).copy()
df["month"] = df[DATE_COL].dt.month
df["night_id"] = df[DATE_COL].dt.strftime("%Y-%m-%d")  # whole-night resampling unit


def boot_ci(stats):
    return np.nanpercentile(stats, [2.5, 97.5])


# =====================================================================
# Q2 Part B: wildfire season vs rest of year, within each site
# (descriptive, so all reports 2018-2025 are used; no model fitting)
# =====================================================================
df["fire_season"] = df["month"].isin(SMOKE_SEASON_MONTHS).astype(int)


def within_site_diff_arr(site_code, season, y):
    """Pooled within-site difference: site-demeaned LM regressed on season flag
    (= weighted average of per-site season-minus-rest differences)."""
    n = np.bincount(site_code).astype(float)
    n[n == 0] = np.nan
    ym = np.bincount(site_code, weights=y) / n
    xm = np.bincount(site_code, weights=season) / n
    yd, xd = y - ym[site_code], season - xm[site_code]
    den = (xd ** 2).sum()
    return np.nan if den == 0 else (xd * yd).sum() / den


# keep only sites that have both seasons with enough reports
cnt = df.groupby([SITE_COL, "fire_season"]).size().unstack(fill_value=0)
cnt = cnt.reindex(columns=[0, 1], fill_value=0)
eligible = cnt[(cnt[0] >= MIN_PER_GROUP) & (cnt[1] >= MIN_PER_GROUP)].index
dq = df[df[SITE_COL].isin(eligible)].copy()
dropped = sorted(set(df[SITE_COL]) - set(eligible))

rows = []
for s, g in dq.groupby(SITE_COL):
    a, b = g[g.fire_season == 1][TARGET], g[g.fire_season == 0][TARGET]
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    rows.append(dict(site=s, n_fire=len(a), n_rest=len(b), mean_fire=a.mean(),
                     mean_rest=b.mean(), diff=diff, ci_low=diff - 1.96 * se,
                     ci_high=diff + 1.96 * se))
q2b_sites = pd.DataFrame(rows).sort_values("diff")

sc = pd.factorize(dq[SITE_COL])[0]
se_ = dq["fire_season"].to_numpy(float)
yy = dq[TARGET].to_numpy(float)
pooled = within_site_diff_arr(sc, se_, yy)
nid_q = dq["night_id"].to_numpy()
idx_q = {k: np.where(nid_q == k)[0] for k in np.unique(nid_q)}
keys_q = list(idx_q)
boots = []
for _ in range(N_BOOT):
    pick = rng.choice(len(keys_q), size=len(keys_q), replace=True)
    ix = np.concatenate([idx_q[keys_q[j]] for j in pick])
    boots.append(within_site_diff_arr(sc[ix], se_[ix], yy[ix]))
lo, hi = boot_ci(boots)
q2b_sites.to_csv(f"{OUT}/q2b_by_site.csv", index=False)

# figure
fig, ax = plt.subplots(figsize=(7, 0.45 * len(q2b_sites) + 1.5))
y = np.arange(len(q2b_sites))
ax.errorbar(q2b_sites["diff"], y,
            xerr=[q2b_sites["diff"] - q2b_sites["ci_low"], q2b_sites["ci_high"] - q2b_sites["diff"]],
            fmt="o", capsize=3)
ax.axvline(0, color="grey", lw=1)
ax.axvline(pooled, color="red", ls="--", lw=1, label=f"pooled {pooled:+.2f}")
ax.set_yticks(y)
ax.set_yticklabels([f"{r.site} (n={r.n_fire}/{r.n_rest})" for r in q2b_sites.itertuples()])
ax.set_xlabel("Limiting magnitude: wildfire season minus rest of year")
ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/q2b_forest.png", dpi=150); plt.close()

# =====================================================================
# Q3: association, same-night conditions, site fixed effects
# =====================================================================
feats = [f for f in Q3_FEATURES if f != SITE_COL]
missing = [f for f in feats if f not in df.columns]
if missing:
    raise KeyError(f"Q3 features missing from reports: {missing}")
# PM2.5 is blank where the nearest monitor is too far away (cleaning report). Dropping those rows
# would remove 128 of Crestone's 202 reports, so fill them with the median of the observed values
# and keep pm25_missing as an indicator: its coefficient absorbs any difference for those rows.
d3 = df.copy()
if "log_pm25_ugm3" in feats:
    d3["log_pm25_ugm3"] = d3["log_pm25_ugm3"].fillna(d3["log_pm25_ugm3"].median())
d3 = d3.dropna(subset=feats + [TARGET])
n_dropped = len(df) - len(d3)

X_num = pd.get_dummies(d3[feats], drop_first=True, dtype=float)
site_d = pd.get_dummies(d3[SITE_COL], prefix="site", drop_first=True, dtype=float)
y3 = d3[TARGET].to_numpy(float)
X = pd.concat([X_num, site_d], axis=1)
X_const = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
names = ["const"] + list(X.columns)
focus = list(X_num.columns)  # report the condition coefficients, not site dummies

beta_hat = np.linalg.lstsq(X_const, y3, rcond=None)[0]
resid = y3 - X_const @ beta_hat
r2 = 1 - resid.var() / y3.var()

nid = d3["night_id"].to_numpy()
idx_by_night = {k: np.where(nid == k)[0] for k in np.unique(nid)}
keys = list(idx_by_night)
B = np.empty((N_BOOT, len(names)))
for i in range(N_BOOT):
    pick = rng.choice(len(keys), size=len(keys), replace=True)
    ix = np.concatenate([idx_by_night[keys[j]] for j in pick])
    B[i] = np.linalg.lstsq(X_const[ix], y3[ix], rcond=None)[0]
ci = np.percentile(B, [2.5, 97.5], axis=0)

sd = X_num.std().to_dict()
rows = []
for c in focus:
    k = names.index(c)
    rows.append(dict(term=c, coef=beta_hat[k], ci_low=ci[0, k], ci_high=ci[1, k],
                     coef_per_sd=beta_hat[k] * sd[c],
                     ci_low_per_sd=ci[0, k] * sd[c], ci_high_per_sd=ci[1, k] * sd[c]))
q3_ols = pd.DataFrame(rows)

# Ordinal logistic check (cumulative logit; same predictors + site dummies)
def fit_ordinal(Xm, yc, start=None):
    """Proportional-odds logit. yc = integer category codes 0..K-1."""
    K = yc.max() + 1
    p = Xm.shape[1]
    def unpack(th):
        beta = th[:p]
        cuts = np.cumsum(np.r_[th[p], np.exp(th[p + 1:])])
        return beta, cuts
    def nll(th):
        beta, cuts = unpack(th)
        eta = Xm @ beta
        c = np.r_[-np.inf, cuts, np.inf]
        pr = expit(c[yc + 1] - eta) - expit(c[yc] - eta)
        return -np.log(np.clip(pr, 1e-12, None)).sum() + 1e-4 * (beta ** 2).sum()
    if start is None:
        start = np.r_[np.zeros(p), np.linspace(-1, 1, K - 1)[:1], np.zeros(K - 2) + 0.0]
    r = minimize(nll, start, method="BFGS", options={"maxiter": 400})
    return r.x, r

q3_ord = None
try:
    Xo = X.loc[:, X.std() > 0]
    foc_o = [c for c in focus if c in Xo.columns]
    Xm = Xo.to_numpy(float)
    # standardize for stable optimisation, convert back afterwards
    mu, sg = Xm.mean(0), Xm.std(0)
    Xz = (Xm - mu) / sg
    # pool sparse levels into the adjacent level so resamples keep every category
    cats = np.sort(np.unique(y3))
    counts = pd.Series(y3).value_counts().reindex(cats).to_numpy()
    grp = np.arange(len(cats))
    MIN_LEVEL = 25
    while True:
        gc = pd.Series(counts).groupby(grp).sum()
        small = gc[gc < MIN_LEVEL]
        if small.empty or len(gc) <= 3:
            break
        g = small.index[0]
        grp[grp == g] = (g - 1 if g > gc.index.min() else gc.index[1])
    yc = pd.Series(grp).map({g: i for i, g in enumerate(sorted(set(grp)))}).to_numpy()[np.searchsorted(cats, y3)]
    cats = np.unique(yc)
    th0, _ = fit_ordinal(Xz, yc)
    p = Xz.shape[1]
    beta0 = th0[:p] / sg
    ob = []
    for _ in range(N_BOOT_ORD):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        ix = np.concatenate([idx_by_night[keys[j]] for j in pick])
        ycb = yc[ix]
        if len(np.unique(ycb)) < len(cats):   # a category vanished; skip rep
            continue
        thb, rb = fit_ordinal(Xz[ix], ycb, start=th0)
        ob.append(thb[:p] / sg)
        if len(ob) % 50 == 0:
            print(f'  ordinal bootstrap {len(ob)}/{N_BOOT_ORD}', flush=True)
    ob = np.array(ob)
    rows = []
    for c in foc_o:
        k = list(Xo.columns).index(c)
        l, h = np.percentile(ob[:, k], [2.5, 97.5])
        rows.append(dict(term=c, log_odds=beta0[k], odds_ratio=np.exp(beta0[k]),
                         ci_low_log_odds=l, ci_high_log_odds=h,
                         same_sign_as_ols=bool(np.sign(beta0[k]) == np.sign(q3_ols.set_index("term").loc[c, "coef"])),
                         n_boot_ok=len(ob)))
    q3_ord = pd.DataFrame(rows)
    q3_ord.to_csv(f"{OUT}/q3_ordinal.csv", index=False)
except Exception as e:
    print("Ordinal model failed:", e)

q3_ols.to_csv(f"{OUT}/q3_ols.csv", index=False)

# coefficient plot (per-SD so conditions are comparable)
fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(q3_ols) + 1.5))
y = np.arange(len(q3_ols))
ax.errorbar(q3_ols["coef_per_sd"], y,
            xerr=[q3_ols["coef_per_sd"] - q3_ols["ci_low_per_sd"],
                  q3_ols["ci_high_per_sd"] - q3_ols["coef_per_sd"]], fmt="o", capsize=3)
ax.axvline(0, color="grey", lw=1)
ax.set_yticks(y); ax.set_yticklabels(q3_ols["term"])
ax.set_xlabel("Change in limiting magnitude per 1 SD\n(site fixed effects, 95% bootstrap CI)")
plt.tight_layout(); plt.savefig(f"{OUT}/q3_coefplot.png", dpi=150); plt.close()

# ---------------- summary ----------------
smoke_cols = [c for c in df.columns if "smoke" in c.lower() and not c.startswith("prev_")]
with open(f"{OUT}/summary.txt", "w") as f:
    f.write("Q2 PART B\n")
    f.write(f"Reports used: {len(dq)} across {dq[SITE_COL].nunique()} sites; "
            f"sites dropped (<{MIN_PER_GROUP} reports in a season group): {dropped}\n")
    f.write(f"Pooled within-site difference (Jun-Sep minus rest): {pooled:+.3f} mag, "
            f"95% bootstrap CI [{lo:+.3f}, {hi:+.3f}] (resampling nights, {N_BOOT} reps)\n\n")
    f.write(q2b_sites.round(3).to_string(index=False) + "\n\nQ3\n")
    f.write(f"n = {len(d3)} reports ({n_dropped} dropped for missing values), "
            f"{d3['night_id'].nunique()} nights, {d3[SITE_COL].nunique()} sites. "
            f"R^2 = {r2:.3f}\n")
    f.write(f"Features: {feats} (+ site fixed effects)\nSmoke-like columns: {smoke_cols}\n\n")
    f.write(q3_ols.round(3).to_string(index=False) + "\n\n")
    if q3_ord is not None:
        f.write("Ordinal logistic check\n" + q3_ord.round(3).to_string(index=False) + "\n")
print(open(f"{OUT}/summary.txt").read())