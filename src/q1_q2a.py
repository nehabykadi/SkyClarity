"""
SkyClarity - Milestone 3, Q1 and Q2 Part A (descriptive analysis of nightly conditions)

Q1  How often are Boulder-area nights clear, smoke-free and moonless at the same time,
    and how does that vary by month and year (2018-2025)?
Q2A How much of the summer clear-sky advantage is lost to smoke?

Run from the repo root, after src/prep.py:
    python src/q1_q2a.py

Input:   data/model_ready/nights_boulder.csv  (12 Boulder-area sites, every night)
Outputs: results/q1_q2a/
    q1_good_night_month_year.csv     good-night rate for each month x year
    q1_good_night_by_month.csv       by month, at 10 / 20 / 30 % cloud thresholds
    q1_good_night_by_year.csv        by year, with 95% bootstrap intervals
    q2a_smoke_loss_by_month.csv      clear vs clear-and-smoke-free nights, by month
    q2a_summer_advantage.csv         summer advantage, the part lost to smoke, 95% intervals
    fig_q1_heatmap.png, fig_q1_sensitivity.png, fig_q2a_smoke_loss.png
    summary.txt                      headline numbers for the write-up

Rates are pooled over the 12 sites: each site-night counts once.

Uncertainty: the 12 sites share weather, and smoke episodes last for weeks (day-to-day
correlation of smoke is about 0.7, and still about 0.3 a month apart). Resampling single
nights would treat each night as independent and give intervals that are too narrow. So
95% intervals use a block bootstrap, stratified by calendar month:
  - 2018-2025 figures: whole months are the blocks. For each calendar month, the 8 years'
    blocks are resampled, so every bootstrap sample keeps one of each month.
  - Single-year figures: 7-day blocks within each month (day 1-7, 8-14, ...), resampled
    within their month. These intervals are still somewhat optimistic for smoke.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep import (CLEAR_CLOUD_MAX, CLEAR_CLOUD_SENSITIVITY, MOONLESS_MAX_ILLUM,  # noqa: E402
                  MOONLESS_MAX_UP_FRAC, OUT as MODEL_READY, WILDFIRE_MONTHS)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/q1_q2a"
SEED = 5502
N_BOOT = 2000
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# chart styling (validated palette: blue / orange, sequential blue ramp)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    n = pd.read_csv(MODEL_READY / "nights_boulder.csv", parse_dates=["night_date"])
    n = n[n.is_smoke_free.notna()].copy()            # cleaned data has none missing; safeguard
    n["is_smoke_free"] = n.is_smoke_free.astype(int)
    n["is_clear_smoke_free"] = n.is_clear * n.is_smoke_free
    return n


def blocks(df, cols, size):
    """Sum cols per block; size="month" (one block per year-month) or "week" (7-day blocks)."""
    key = df.year * 100 + df.month
    if size == "week":
        key = key * 10 + (df.night_date.dt.day - 1) // 7
    g = df.groupby(key)
    t = g[cols].sum()
    t["k"] = g.size()
    t["month"] = g.month.first()
    return t.reset_index(drop=True)


def boot(t, stat, rng):
    """95% interval of stat(table), resampling blocks with replacement within each calendar month."""
    strata = [np.flatnonzero(t.month.to_numpy() == m) for m in sorted(t.month.unique())]
    vals = [stat(t.iloc[np.concatenate([rng.choice(g, len(g)) for g in strata])]) for _ in range(N_BOOT)]
    return np.nanpercentile(np.array(vals, dtype=float), [2.5, 97.5], axis=0)


def rate_ci(df, col, size, rng):
    return boot(blocks(df, [col], size), lambda b: b[col].sum() / b.k.sum(), rng)


def q1(n, rng):
    rate = lambda df, col="is_good_night": df[col].mean()

    my = n.pivot_table(index="month", columns="year", values="is_good_night", aggfunc="mean")
    my.to_csv(RESULTS / "q1_good_night_month_year.csv", float_format="%.4f")

    cols = ["is_good_night"] + [f"is_good_night_cloud{t}" for t in CLEAR_CLOUD_SENSITIVITY]
    bm = n.groupby("month")[cols + ["is_clear", "is_moonless", "is_smoke_free"]].mean()
    bm = bm.rename(columns={"is_good_night": f"is_good_night_cloud{CLEAR_CLOUD_MAX}"})
    bm.to_csv(RESULTS / "q1_good_night_by_month.csv", float_format="%.4f")

    rows = []
    for y, g in n.groupby("year"):
        lo, hi = rate_ci(g, "is_good_night", "week", rng)
        rows.append({"year": y, "good_night_rate": rate(g), "ci_low": lo, "ci_high": hi,
                     "clear_rate": rate(g, "is_clear"), "smoke_free_rate": rate(g, "is_smoke_free")})
    by = pd.DataFrame(rows)
    by.to_csv(RESULTS / "q1_good_night_by_year.csv", index=False, float_format="%.4f")

    # heatmap: month x year
    fig, ax = plt.subplots(figsize=(8, 5.6))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vmax = np.ceil(my.values.max() * 20) / 20
    im = ax.imshow(my.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(my.shape[0]):
        for j in range(my.shape[1]):
            v = my.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > vmax * 0.55 else INK)
    ax.set_xticks(range(my.shape[1]), my.columns)
    ax.set_yticks(range(12), MONTHS)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=matplotlib.ticker.PercentFormatter(1, 0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, length=0)
    ax.set_title("Share of Boulder-area site-nights that are clear, smoke-free and moonless")
    fig.text(0.01, 0.01, f"12 sites, 2018–2025. Clear: mean cloud ≤ {CLEAR_CLOUD_MAX}% over the dark hours; "
             f"smoke-free: no NOAA HMS plume; moonless: moon up ≤ {MOONLESS_MAX_UP_FRAC:.0%} of the night "
             f"or ≤ {MOONLESS_MAX_ILLUM:.0%} lit.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "fig_q1_heatmap.png", dpi=200)
    plt.close(fig)

    # sensitivity: good-night rate by month at three cloud thresholds (ordered -> one hue, 3 steps)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(1, 13)
    for t, c in zip(sorted(CLEAR_CLOUD_SENSITIVITY + [CLEAR_CLOUD_MAX]), ["#86b6ef", BLUE, "#104281"]):
        y = bm[f"is_good_night_cloud{t}"].to_numpy()
        ax.plot(x, y, color=c, lw=2, marker="o", ms=4)
        ax.annotate(f"cloud ≤ {t}%", (12, y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_xticks(x, MONTHS)
    ax.set_xlim(0.6, 13.6)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel("Good-night rate")
    ax.set_title("The monthly pattern holds across cloud thresholds")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_q1_sensitivity.png", dpi=200)
    plt.close(fig)
    return my, bm, by


def q2a(n, rng):
    # "Clear sky" here uses cloud only: the moon cycle is the same in every season,
    # so leaving it out isolates the seasonal cloud advantage and the smoke loss.
    bm = n.groupby("month").agg(clear=("is_clear", "mean"),
                                clear_smoke_free=("is_clear_smoke_free", "mean"),
                                clear_moonless=("is_clear_moonless", "mean"),
                                good_night=("is_good_night", "mean"),
                                dark_hours=("n_dark_hours", "mean"))
    bm["lost_to_smoke"] = bm.clear - bm.clear_smoke_free
    bm["share_of_clear_nights_smoky"] = bm.lost_to_smoke / bm.clear
    bm["clear_dark_hours"] = bm.clear * bm.dark_hours
    bm.to_csv(RESULTS / "q2a_smoke_loss_by_month.csv", float_format="%.4f")

    def stats(df):
        s = df.month.isin(WILDFIRE_MONTHS)
        adv = df.is_clear[s].mean() - df.is_clear[~s].mean()
        adv_sf = df.is_clear_smoke_free[s].mean() - df.is_clear_smoke_free[~s].mean()
        return adv, adv_sf, (adv - adv_sf) / adv if adv > 0 else np.nan

    def boot_stats(df, size):
        def stat(b):
            sm = b.month.isin(WILDFIRE_MONTHS)
            adv = b.is_clear[sm].sum() / b.k[sm].sum() - b.is_clear[~sm].sum() / b.k[~sm].sum()
            adv_sf = (b.is_clear_smoke_free[sm].sum() / b.k[sm].sum()
                      - b.is_clear_smoke_free[~sm].sum() / b.k[~sm].sum())
            return adv, adv_sf, (adv - adv_sf) / adv if adv > 0 else np.nan
        return boot(blocks(df, ["is_clear", "is_clear_smoke_free"], size), stat, rng)

    rows = []
    for label, df in [("2018-2025", n)] + [(str(y), g) for y, g in n.groupby("year")]:
        adv, adv_sf, lost = stats(df)
        ci = boot_stats(df, "month" if label == "2018-2025" else "week")
        rows.append({"period": label, "summer_clear_rate": df.is_clear[df.month.isin(WILDFIRE_MONTHS)].mean(),
                     "rest_clear_rate": df.is_clear[~df.month.isin(WILDFIRE_MONTHS)].mean(),
                     "advantage": adv, "advantage_ci_low": ci[0, 0], "advantage_ci_high": ci[1, 0],
                     "advantage_smoke_free": adv_sf, "share_lost_to_smoke": lost,
                     "share_lost_ci_low": ci[0, 2], "share_lost_ci_high": ci[1, 2]})
    sa = pd.DataFrame(rows)
    sa.to_csv(RESULTS / "q2a_summer_advantage.csv", index=False, float_format="%.4f")

    # figure: clear vs clear-and-smoke-free by month, gap = lost to smoke
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(1, 13)
    ax.axvspan(min(WILDFIRE_MONTHS) - 0.5, max(WILDFIRE_MONTHS) + 0.5, color="#f1f0ec", zorder=0)
    ax.text(np.mean(WILDFIRE_MONTHS), 0.015, "wildfire season", ha="center", fontsize=8.5, color=INK2)
    ax.fill_between(x, bm.clear_smoke_free, bm.clear, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(x, bm.clear, color=BLUE, lw=2, marker="o", ms=4)
    ax.plot(x, bm.clear_smoke_free, color=ORANGE, lw=2, marker="o", ms=4)
    top, low = bm.clear.idxmax(), bm.clear_smoke_free[bm.index.isin(WILDFIRE_MONTHS)].idxmin()
    ax.annotate("clear", (top, bm.clear[top]), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.annotate("clear and smoke-free", (low, bm.clear_smoke_free[low]), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=9, color=ORANGE, fontweight="bold")
    peak = bm.lost_to_smoke.idxmax()
    ax.annotate(f"{bm.lost_to_smoke[peak]:.0%} of {MONTHS[peak - 1]} nights\nare clear but smoky",
                (peak, (bm.clear[peak] + bm.clear_smoke_free[peak]) / 2), xytext=(3.0, 0.44),
                textcoords="data", ha="center", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xticks(x, MONTHS)
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel("Share of site-nights")
    ax.set_title("Smoke cancels most of summer's clear-sky advantage")
    fig.text(0.01, 0.01, f"12 Boulder-area sites, 2018–2025. Clear: mean cloud ≤ {CLEAR_CLOUD_MAX}% over the dark hours. "
             "Shaded gap: clear nights with a NOAA HMS smoke plume overhead.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "fig_q2a_smoke_loss.png", dpi=200)
    plt.close(fig)
    return bm, sa


def smoke_sensitivity(n):
    """Share of the summer advantage lost if light smoke (HMS level 1) is treated as smoke-free."""
    s = n.month.isin(WILDFIRE_MONTHS)
    csf = n.is_clear * (n.hms_smoke_level <= 1)
    adv = n.is_clear[s].mean() - n.is_clear[~s].mean()
    return (adv - (csf[s].mean() - csf[~s].mean())) / adv


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n = load()
    my, bm1, by = q1(n, rng)
    bm2, sa = q2a(n, rng)

    lo, hi = rate_ci(n, "is_good_night", "month", rng)
    allp = sa.iloc[0]
    best, worst = bm1[f"is_good_night_cloud{CLEAR_CLOUD_MAX}"].idxmax(), bm1[f"is_good_night_cloud{CLEAR_CLOUD_MAX}"].idxmin()
    lines = [
        "SkyClarity - Q1 and Q2 Part A results", f"seed {SEED}, {N_BOOT} block-bootstrap resamples (month blocks for 2018-2025, 7-day blocks within a year)", "",
        "Q1  Good nights (clear, smoke-free and moonless)",
        f"  Overall: {n.is_good_night.mean():.1%} of site-nights (95% CI {lo:.1%}-{hi:.1%}), "
        f"{len(n):,} site-nights, {n.night_date.nunique():,} nights",
        f"  Best month {MONTHS[best - 1]} {bm1.loc[best, f'is_good_night_cloud{CLEAR_CLOUD_MAX}']:.1%}, "
        f"worst month {MONTHS[worst - 1]} {bm1.loc[worst, f'is_good_night_cloud{CLEAR_CLOUD_MAX}']:.1%}",
        "  By year:",
        "  " + by.assign(rate=by.good_night_rate.map("{:.1%}".format),
                         ci=by.ci_low.map("{:.1%}".format) + "-" + by.ci_high.map("{:.1%}".format))
              [["year", "rate", "ci"]].to_string(index=False).replace("\n", "\n  "),
        "  Sensitivity (overall rate): " + ", ".join(
            f"cloud <= {t}%: {n[f'is_good_night_cloud{t}' if t != CLEAR_CLOUD_MAX else 'is_good_night'].mean():.1%}"
            for t in sorted(CLEAR_CLOUD_SENSITIVITY + [CLEAR_CLOUD_MAX])),
        "",
        "Q2A  Summer clear-sky advantage and the part lost to smoke (June-September vs rest of year)",
        f"  Clear nights: summer {allp.summer_clear_rate:.1%} vs rest {allp.rest_clear_rate:.1%}",
        f"  Advantage: {allp.advantage * 100:+.1f} points (95% CI {allp.advantage_ci_low * 100:+.1f} to "
        f"{allp.advantage_ci_high * 100:+.1f})",
        f"  Advantage once smoky nights are removed: {allp.advantage_smoke_free * 100:+.1f} points",
        f"  Share of the advantage lost to smoke: {allp.share_lost_to_smoke:.0%} "
        f"(95% CI {allp.share_lost_ci_low:.0%}-{allp.share_lost_ci_high:.0%})",
        "  By year:",
        "  " + sa.iloc[1:].assign(adv=(sa.advantage * 100).map("{:+.1f}".format),
                                  lost=sa.share_lost_to_smoke.map(lambda v: "n/a" if pd.isna(v) else f"{v:.0%}"))
              [["period", "adv", "lost"]].to_string(index=False).replace("\n", "\n  "),
        "  Sensitivity, counting only medium or heavy smoke as smoky: "
        f"{smoke_sensitivity(n):.0%} of the advantage lost",
        "  Share of clear nights with smoke overhead, by month:",
        "  " + ", ".join(f"{MONTHS[m - 1]} {v:.0%}" for m, v in bm2.share_of_clear_nights_smoky.items()),
        "",
        "Caveats: HMS plumes are drawn from daytime imagery; weather is reanalysis, not site observation;",
        "the 12 sites share weather, so they are not 12 independent samples.",
    ]
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()