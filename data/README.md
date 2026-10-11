# Data — Astronomy Sky Clarity

This folder holds two analysis-ready tables, the raw files they were built from, and a log of every download. Collection is finished; ready for cleaning and preprocessing.

- **Built by:** Daralynn Rhode, with `build_dataset.py`
- **Last build:** 2026-10-11
- **Study period:** 2018-01-01 to 2025-12-31
- **Area:** 16 sites in the Colorado Front Range (Boulder area, plus Denver and Crestone)

## The two datasets at a glance

| | Dataset 1 | Dataset 2 |
|---|---|---|
| File | `processed/dataset1_site_night.csv` | `processed/dataset2_gan_observations.csv` |
| One row is | one site on one night | one Globe at Night observer report |
| Rows | 46,768 (16 sites × 2,923 nights) | 3,579 (2,884 matched to Dataset 1) |
| Columns | 28 | 38 |
| Key | `site_id` + `night_date` (unique) | `gan_id`; joins to Dataset 1 on `site_id` + `night_date` |
| Role | Conditions: weather, moon, smoke, light pollution | Outcome: how dark the sky looked, with that night's conditions attached |

Start with **Dataset 2** for anything that predicts sky quality. Use **Dataset 1** for questions about conditions alone, such as how often smoke coincides with clear, moonless nights.

## Folder layout

```
data/
├── README.md                          this file
├── processed/
│   ├── dataset1_site_night.csv
│   ├── dataset2_gan_observations.csv
│   ├── build_report.txt               row counts, missing-value rates, match rates from the last build
│   ├── _hms_site_daily.csv            saved intermediate: smoke level per site per day
│   └── _viirs_site_year.csv           saved intermediate: radiance per site per year
└── raw/
    ├── manifest.csv                   every automatic download: URL, UTC time, bytes, SHA-256
    ├── sites.csv                      the 16 sites and their coordinates
    ├── open_meteo/                    hourly weather, one JSON per site per year (128 files)
    ├── aqs/                           EPA daily PM2.5, national yearly zips (16 files)
    ├── hms/                           NOAA smoke polygons, one zip per day (2,922 files)
    ├── globe_at_night/                yearly CSVs, downloaded by hand (8 files)
    ├── viirs/                         annual night-lights GeoTIFFs, downloaded by hand (8 files, ~12 GB each)
    └── ephemeris/                     JPL DE421 file used for sun and moon positions
```

> `raw/` is too large for GitHub. If it is not in the repo, it is in the team's shared folder on Google Drive: https://drive.google.com/file/d/1NNblHwz1sJzswq5HCODqq2127qxmp_jP/view?usp=sharing. You only need `raw/` to trace a value back to its source or to rebuild.

## Dataset 1: nightly conditions

**What it covers.** Every night from 2018 through 2025 at each of the 16 sites, whether or not anyone observed the sky that night.

**What counts as a night.** The hours when the sun is more than 18° below the horizon (astronomical darkness). `night_date` is the local (America/Denver) calendar date on which that evening began, so the early hours of 15 August belong to the night of 14 August.

**How it was put together.**

1. **Weather.** Hourly values for each site were pulled from the Open-Meteo Historical Weather API (modelled reanalysis, not station observations).
2. **Sun and moon.** For every site-hour, sun altitude, moon altitude and moon illuminated fraction were computed with the `skyfield` library.
3. **Collapse to nights.** Only the dark hours were kept and summarised per site and night: means for cloud, humidity, temperature and wind; the maximum for cloud and gusts; the sum for precipitation; a circular mean for wind direction.
4. **PM2.5.** Each site-day takes the daily mean from the nearest EPA monitor within 75 km that reported that day. The monitor used and its distance are recorded on every row. Wildfire-flagged readings are kept.
5. **Smoke.** Each site was tested against that day's NOAA smoke plume polygons. The densest plume over the site gives `hms_smoke_level` (0 none, 1 light, 2 medium, 3 heavy). Plumes are drawn from daytime imagery, so a night gets the smoke seen earlier that day.
6. **Light pollution.** The VIIRS annual radiance at the site's pixel was joined by calendar year. Years 2018–2021 are product version V2.1 and 2022–2025 are V2.2.

**Columns.**

| Column | Meaning | Units |
|---|---|---|
| `site_id`, `site_name` | Site identifier and readable name | |
| `latitude`, `longitude`, `elevation_m` | Site location | degrees, metres |
| `night_date` | Local date the evening began | YYYY-MM-DD |
| `n_dark_hours` | Hours of astronomical darkness | 5 to 11 |
| `cloud_cover_mean`, `cloud_cover_max` | Total cloud cover over the dark hours | % of sky |
| `cloud_low_mean`, `cloud_mid_mean`, `cloud_high_mean` | Cloud cover by layer (below 2 km, 2–6 km, above 6 km) | % of sky |
| `rel_humidity_mean` | Relative humidity at 2 m | % |
| `temp_c_mean`, `dew_point_c_mean` | Air temperature and dew point at 2 m | °C |
| `precip_mm_sum` | Precipitation over the dark hours | mm |
| `wind_speed_ms_mean`, `wind_gust_ms_max` | Wind speed and peak gust at 10 m | m/s |
| `wind_dir_deg_mean` | Wind direction, circular mean | degrees, 0 = from north |
| `moon_illum_frac` | Illuminated fraction of the moon | 0 to 1 |
| `moon_up_frac` | Share of dark hours with the moon above the horizon | 0 to 1 |
| `moon_alt_max_deg` | Highest moon altitude in the dark hours (negative = never rose) | degrees |
| `pm25_ugm3` | Daily mean PM2.5 on the calendar day of `night_date` | µg/m³ |
| `pm25_next_day_ugm3` | Daily mean PM2.5 on the following day | µg/m³ |
| `pm25_monitor_id`, `pm25_monitor_km` | EPA monitor used and its distance from the site | code, km |
| `hms_smoke_level` | Densest smoke plume over the site that day | 0, 1, 2, 3 |
| `viirs_radiance` | Annual mean night-time radiance at the site's pixel | nW/cm²/sr |

## Dataset 2: sky quality observations

**What it covers.** Every Globe at Night report from 2018–2025 made inside a Colorado bounding box (latitude 36.9 to 41.1, longitude −109.1 to −102.0). Globe at Night is a citizen-science programme in which observers compare the sky to star charts.

**The outcome variables.**

- `limiting_mag` — faintest stars visible to the naked eye, 0 (worst) to 7 (best). Primary outcome; present for 2,691 matched reports.
- `sqm_reading` — sky quality meter reading in mag/arcsec². Secondary outcome; present for 414 matched reports.

**How it was put together.**

1. The eight yearly CSVs were stacked and filtered to the bounding box.
2. Limiting magnitudes below 0 and meter readings of 0 or less were set to blank. Nothing else was altered.
3. `night_date` was derived as the report's local date and time minus 12 hours, so a report at 01:00 belongs to the previous evening.
4. Each report was assigned to its nearest site. If that site is within 25 km, the report is matched to the Dataset 1 row with the same `site_id` and `night_date`, and that row's conditions are copied on.
5. Reports farther than 25 km from every site are **kept**, with `matched = False` and blank condition columns.

**Columns specific to Dataset 2.** All Dataset 1 columns from `elevation_m` through `viirs_radiance` are also present.

| Column | Meaning |
|---|---|
| `gan_id` | Globe at Night record identifier |
| `gan_source_file` | Yearly file the record came from |
| `obs_local_time` | Local date and time as entered by the observer |
| `night_date` | Night the report belongs to |
| `obs_latitude`, `obs_longitude`, `obs_elevation_m` | Observer location |
| `limiting_mag` | Naked-eye limiting magnitude, 0 to 7 |
| `sqm_reading` | Sky quality meter reading |
| `observer_cloud_cover` | Cloud cover judged by the observer: clear, 1/4 of sky, 1/2 of sky, over 1/2 of sky |
| `constellation` | Constellation used for the estimate |
| `site_id` | Nearest project site |
| `site_km` | Distance from the observer to that site |
| `within_radius` | True if `site_km` is 25 km or less |
| `matched` | True if Dataset 1 conditions were attached |

**Matched reports per site.**

| Site | Reports | Site | Reports |
|---|---|---|---|
| sommers_bausch | 803 | nederland | 66 |
| boulder_downtown | 523 | lyons | 28 |
| denver_downtown | 481 | golden_gate_sp | 28 |
| flagstaff_mtn | 218 | brainard_lake | 10 |
| crestone | 202 | rmnp_alpine_vc | 10 |
| gross_reservoir | 192 | jackson_lake_sp | 4 |
| longmont | 177 | estes_park | 3 |
| ncar_mesa | 137 | pawnee_buttes | 2 |

## What is done and what is left

Nothing was imputed and no rows were dropped during the build. Every cleaning decision is still open.

| Already done | Left for cleaning and preprocessing |
|---|---|
| Hourly weather collapsed to one row per site-night | Decide how to handle missing PM2.5 |
| All sources joined on site and night | Outlier and range checks |
| Invalid limiting magnitudes and zero meter readings blanked | Filter unreliable observer reports |
| Unmatched reports flagged, not removed | Encoding, scaling and derived features |

## Known issues to handle

| Issue | Detail | Where |
|---|---|---|
| Rows before the study period | 16 rows dated 2017-12-31, one per site, with no smoke or radiance. Safe to drop. | Dataset 1 |
| Missing PM2.5 | 9.38% of rows. Pawnee Buttes has no monitor within 75 km; Crestone is missing on about half of nights. | Dataset 1 |
| Negative PM2.5 | 9 rows, lowest −1.03. This is instrument noise in the EPA data. | Dataset 1 |
| Distant PM2.5 monitors | RMNP ~59 km, Jackson Lake ~52 km, Estes Park ~43 km. Use `pm25_monitor_km` to filter or weight. | Dataset 1 |
| Zero radiance | Four remote sites read 0. This means below the detection floor, not missing. | Dataset 1 |
| Gross Reservoir radiance | Rises from about 1 (2018–2021) to 25 in 2024 and 50 in 2025. Cause not confirmed. | Dataset 1 |
| Radiance version change | V2.1 through 2021, V2.2 from 2022. Downtown Boulder and Longmont peak in 2022, so some of that rise may be the version change. | Dataset 1 |
| Daytime timestamps | 339 matched reports have a local time between 06:00 and 18:00, so the time was probably entered wrong. | Dataset 2 |
| Implausible meter readings | 52 of 414 matched readings fall outside 14 to 23. | Dataset 2 |
| Unmatched reports | 695 rows with `matched = False`. Filter these out before modelling. | Dataset 2 |

## Limitations to keep in mind

- **Smoke is rare in the outcome data.** Only 136 of the 2,884 matched reports fall on a day with smoke overhead (98 light, 32 medium, 6 heavy). Smoke peaks in June to September, but only 318 matched reports come from those months. Any estimate of the smoke effect will be uncertain and tangled up with season.
- **Reports are concentrated.** Sommers-Bausch, downtown Boulder and downtown Denver hold 63% of matched reports.
- **Boulder-area sites are not independent.** Several share the same PM2.5 monitor and the same weather grid cell, so they differ mainly in radiance and elevation.
- **Crestone covers different years.** Nearly all of its reports are from 2022 and 2023.
- **Weather is modelled.** Cloud cover comes from a reanalysis grid, not from observations at the site.

A sanity check that came out well: mean PM2.5 rises with satellite smoke level (5.9, 8.0, 15.3 and 22.6 µg/m³ for none, light, medium, heavy). The two sources are independent, so they support each other.

## Rebuilding

You do not need to rebuild to start cleaning. If you change the site list, the date range or a join distance (all set at the top of `build_dataset.py`):

```
pip install pandas numpy requests geopandas skyfield rasterio
python build_dataset.py download     # fetches anything not already in data/raw/
python build_dataset.py combine      # rebuilds both tables and build_report.txt
```

- Downloads are cached, so re-running only fetches what is missing.
- Globe at Night CSVs and VIIRS GeoTIFFs must be downloaded by hand into `raw/globe_at_night/` and `raw/viirs/`.
- The first `combine` after a site-list change takes longer, because smoke and radiance are recomputed for every site.

## Sources

| Source | Provider | Where |
|---|---|---|
| Globe at Night observations | NSF NOIRLab | https://globeatnight.org/maps-data/ |
| Historical Weather API | Open-Meteo | https://open-meteo.com/en/docs/historical-weather-api |
| AQS daily PM2.5 (parameters 88101, 88502) | US EPA | https://aqs.epa.gov/aqsweb/airdata/download_files.html |
| HMS smoke polygons | NOAA NESDIS | https://www.ospo.noaa.gov/products/land/hms.html |
| VIIRS Annual Nighttime Lights V2.1 and V2.2 | Earth Observation Group, Colorado School of Mines | https://eogdata.mines.edu/products/vnl/ |
| Sun and moon positions | `skyfield` with JPL DE421 | computed locally |

Exact URLs, access times and checksums for every automatic download are in `raw/manifest.csv`. Full provenance and the data dictionary are in the project's provenance document.
