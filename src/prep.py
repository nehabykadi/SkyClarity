"""
SkyClarity - Milestone 3 shared preparation (Step 0)

Turns the cleaned Milestone 2 datasets into the model-ready tables that all four
research questions use, so every team member works from identical filters,
flags, features and splits.

Run from the repo root:
    python src/prep.py

Inputs (from Milestone 2 cleaning):
    data/processed/dataset1_site_night_cleaned.csv
    data/processed/dataset2_gan_observations_cleaned.csv

Outputs (data/model_ready/):
    nights_boulder.csv   Q1, Q2 Part A   one row = one Boulder-area site on one night,
                                         with clear / smoke-free / moonless / good-night flags
    reports_model.csv    Q2 Part B, Q3, Q4   one row = one cleaned observer report, with
                                         previous-night (lag) features and the train/test split
    prep_summary.txt     counts per split and site, flag rates, settings used

Other scripts can also import the shared definitions instead of re-typing them:
    from prep import Q3_FEATURES, Q4_FEATURES, TARGET, load_reports
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings (agree as a team)
ROOT = Path(__file__).resolve().parent.parent
NIGHTS_IN = ROOT / "data/processed/dataset1_site_night_cleaned.csv"
REPORTS_IN = ROOT / "data/processed/dataset2_gan_observations_cleaned.csv"
OUT = ROOT / "data/model_ready"

# "Boulder area" for Q1 and Q2 Part A. Denver, Crestone, Jackson Lake and Pawnee Buttes are
# outside it. Q2 Part B, Q3 and Q4 use all sites, because site is accounted for there.
BOULDER_SITES = [
    "boulder_downtown", "sommers_bausch", "ncar_mesa", "flagstaff_mtn", "gross_reservoir",
    "nederland", "brainard_lake", "lyons", "longmont", "estes_park", "rmnp_alpine_vc",
    "golden_gate_sp",
]

# Good-night definition (Q1). The main threshold plus two for the sensitivity check.
CLEAR_CLOUD_MAX = 20             # mean cloud cover over the dark hours, %
CLEAR_CLOUD_SENSITIVITY = [10, 30]
SMOKE_FREE_MAX_LEVEL = 0         # HMS smoke level: 0 none, 1 light, 2 medium, 3 heavy
MOONLESS_MAX_UP_FRAC = 0.25      # moonless = moon up for at most 25% of the dark hours ...
MOONLESS_MAX_ILLUM = 0.25        # ... OR at most 25% illuminated

WILDFIRE_MONTHS = [6, 7, 8, 9]   # June-September

# Time split for Q4 (and any held-out check). Never tune on the test years.
TRAIN_YEARS = range(2018, 2023)  # 2018-2022
TEST_YEARS = range(2023, 2026)   # 2023-2025

TARGET = "limiting_mag"

# Columns lagged by one night within each site: "what was known the day before".
LAG_COLUMNS = [
    "cloud_cover_mean", "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
    "rel_humidity_mean", "dew_point_c_mean", "precip_mm_sum", "wind_speed_ms_mean",
    "pm25_ugm3", "pm25_missing", "hms_smoke_level",
]

# Q3: same-night conditions, with site as fixed effects (add site dummies in the model).
Q3_FEATURES = ["cloud_cover_mean", "hms_smoke_level", "log_pm25_ugm3", "pm25_missing",
               "moon_illum_frac", "moon_up_frac"]

# Q4: ONLY information known the day before. The moon and darkness are predictable in advance.
Q4_FEATURES = (
    ["moon_illum_frac", "moon_up_frac", "moon_alt_max_deg", "n_dark_hours",
     "log_viirs_radiance", "elevation_m", "month_sin", "month_cos"]
    + [f"prev_{c}" for c in LAG_COLUMNS]
)
# Never use these in Q4: they describe the night being predicted, or the outcome itself.
Q4_EXCLUDED = ["cloud_cover_mean", "cloud_cover_max", "cloud_low_mean", "cloud_mid_mean",
               "cloud_high_mean", "rel_humidity_mean", "temp_c_mean", "dew_point_c_mean",
               "precip_mm_sum", "log_precip_mm_sum", "wind_speed_ms_mean", "wind_gust_ms_max",
               "wind_dir_deg_mean", "pm25_ugm3", "log_pm25_ugm3", "pm25_missing",
               "hms_smoke_level", "observer_cloud_cover", "sqm_reading"]

# Observer coordinates are left out of every output (cleaning report section 3.1).
PRIVATE_COLUMNS = ["obs_latitude", "obs_longitude", "obs_elevation_m"]


# ---------------------------------------------------------------- building blocks
def load_nights():
    return pd.read_csv(NIGHTS_IN, parse_dates=["night_date"])


def add_calendar(df):
    df = df.copy()
    df["year"] = df.night_date.dt.year
    df["month"] = df.night_date.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    df["wildfire_season"] = df.month.isin(WILDFIRE_MONTHS).astype(int)
    return df


def add_night_flags(df):
    """Clear, smoke-free, moonless and good-night flags (1/0). Smoke-free is blank if smoke is."""
    df = df.copy()
    df["is_clear"] = (df.cloud_cover_mean <= CLEAR_CLOUD_MAX).astype(int)
    df["is_smoke_free"] = (df.hms_smoke_level <= SMOKE_FREE_MAX_LEVEL).astype("Int64")
    df.loc[df.hms_smoke_level.isna(), "is_smoke_free"] = pd.NA
    df["is_moonless"] = ((df.moon_up_frac <= MOONLESS_MAX_UP_FRAC)
                         | (df.moon_illum_frac <= MOONLESS_MAX_ILLUM)).astype(int)
    df["is_clear_moonless"] = df.is_clear * df.is_moonless
    df["is_good_night"] = df.is_clear_moonless * df.is_smoke_free
    for t in CLEAR_CLOUD_SENSITIVITY:
        df[f"is_good_night_cloud{t}"] = ((df.cloud_cover_mean <= t).astype(int)
                                         * df.is_moonless * df.is_smoke_free)
    return df


def lag_features(nights):
    """Previous night's conditions for every site-night, joined by date (not row order)."""
    prev = nights[["site_id", "night_date"] + LAG_COLUMNS].copy()
    prev["night_date"] = prev.night_date + pd.Timedelta(days=1)   # yesterday's row -> today
    return prev.rename(columns={c: f"prev_{c}" for c in LAG_COLUMNS})


def load_reports():
    """Model-ready reports, for other scripts: pd.DataFrame with a 'split' column."""
    return pd.read_csv(OUT / "reports_model.csv", parse_dates=["night_date"])


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nights = load_nights()
    assert not nights.duplicated(["site_id", "night_date"]).any()

    # Q1 / Q2 Part A table
    nb = add_night_flags(add_calendar(nights[nights.site_id.isin(BOULDER_SITES)]))
    missing = set(BOULDER_SITES) - set(nb.site_id)
    assert not missing, f"Boulder sites not found in data: {missing}"
    nb.to_csv(OUT / "nights_boulder.csv", index=False, date_format="%Y-%m-%d")

    # Q2 Part B / Q3 / Q4 table
    r = pd.read_csv(REPORTS_IN, parse_dates=["night_date"])
    n_in = len(r)
    r = r.drop(columns=[c for c in PRIVATE_COLUMNS if c in r.columns])
    r = add_calendar(r).merge(lag_features(nights), on=["site_id", "night_date"], how="left")
    assert len(r) == n_in, "lag join changed the number of reports"
    r["split"] = np.where(r.year.isin(TRAIN_YEARS), "train",
                          np.where(r.year.isin(TEST_YEARS), "test", "unused"))
    r["cv_group"] = r.year            # group cross-validation folds by year, inside train only
    r.to_csv(OUT / "reports_model.csv", index=False, date_format="%Y-%m-%d")

    # summary
    lines = ["SkyClarity Milestone 3 prep summary", "",
             f"Clear: cloud <= {CLEAR_CLOUD_MAX}% (sensitivity {CLEAR_CLOUD_SENSITIVITY}); "
             f"smoke-free: HMS level <= {SMOKE_FREE_MAX_LEVEL}; moonless: moon up <= "
             f"{MOONLESS_MAX_UP_FRAC:.0%} of dark hours OR <= {MOONLESS_MAX_ILLUM:.0%} lit",
             f"Train {TRAIN_YEARS.start}-{TRAIN_YEARS.stop - 1}, test {TEST_YEARS.start}-{TEST_YEARS.stop - 1}",
             "",
             f"nights_boulder.csv: {len(nb):,} rows, {nb.site_id.nunique()} sites",
             f"  clear {nb.is_clear.mean():.1%}, smoke-free {nb.is_smoke_free.mean():.1%}, "
             f"moonless {nb.is_moonless.mean():.1%}, good night {nb.is_good_night.mean():.1%}",
             "  good-night rate by month:",
             "  " + nb.groupby("month").is_good_night.mean().map("{:.1%}".format).to_string().replace("\n", "\n  "),
             "",
             f"reports_model.csv: {len(r):,} rows",
             "  rows missing previous-night features: "
             f"{int(r.prev_cloud_cover_mean.isna().sum())}",
             "  reports per split:",
             "  " + r.split.value_counts().to_string().replace("\n", "\n  "),
             "  reports per site and split:",
             "  " + pd.crosstab(r.site_id, r.split).to_string().replace("\n", "\n  "),
             "  smoke-day reports per split (HMS level > 0):",
             "  " + r[r.hms_smoke_level > 0].split.value_counts().to_string().replace("\n", "\n  ")]
    (OUT / "prep_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()