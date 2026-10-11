#!/usr/bin/env python3
"""
Astronomy Sky Clarity - data acquisition and preprocessing (CSCI 5502 group project)

Usage
-----
    pip install pandas numpy requests geopandas skyfield rasterio
    python build_dataset.py download     # stage 1: fetch raw files into data/raw/
    python build_dataset.py combine      # stage 2: build the two datasets in data/processed/
    python build_dataset.py all          # both
    (optional)  --start 2018-01-01 --end 2025-12-31

Outputs
-------
    data/raw/manifest.csv                        every downloaded file: URL, UTC access time, size, sha256
    data/processed/dataset1_site_night.csv       one row = one site on one night
    data/processed/dataset2_gan_observations.csv one row = one Globe at Night observation, joined to dataset 1
    data/processed/build_report.txt              row counts, missingness, join match rates

Manual inputs (not downloadable by script)
------------------------------------------
    data/raw/globe_at_night/*.csv   yearly CSVs from https://globeatnight.org/maps-data/
    data/raw/viirs/*.tif            (optional) EOG VIIRS annual VNL V2.x "average_masked" GeoTIFFs,
                                    year must appear in the file name as _YYYY_ ; needs a free EOG login

Downloads are cached: re-running skips files that already exist.
"""
import argparse, hashlib, io, json, re, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------------- config
START, END = "2018-01-01", "2025-12-31"
LOCAL_TZ = "America/Denver"
DARK_SUN_ALT = -18.0          # degrees; astronomical darkness
PM25_MAX_KM = 75              # max distance from a site to an EPA monitor
GAN_MAX_KM = 25               # max distance from a Globe at Night report to a site
GAN_BBOX = (36.9, 41.1, -109.1, -102.0)   # Colorado: lat_min, lat_max, lon_min, lon_max

# site_id, name, latitude, longitude  -- coordinates are approximate: CHECK AND EDIT
SITES = [
    ("boulder_downtown", "Boulder (downtown)",            40.0150, -105.2705),
    ("sommers_bausch",   "Sommers-Bausch Observatory",    40.0037, -105.2625),
    ("ncar_mesa",        "NCAR Mesa Lab",                 39.9780, -105.2750),
    ("flagstaff_mtn",    "Flagstaff Mountain",            40.0020, -105.3070),
    ("gross_reservoir",  "Gross Reservoir",               39.9480, -105.3600),
    ("nederland",        "Nederland",                     39.9614, -105.5108),
    ("brainard_lake",    "Brainard Lake",                 40.0780, -105.5730),
    ("lyons",            "Lyons",                         40.2247, -105.2714),
    ("longmont",         "Longmont",                      40.1672, -105.1019),
    ("estes_park",       "Estes Park",                    40.3772, -105.5217),
    ("rmnp_alpine_vc",   "RMNP Alpine Visitor Center",    40.4410, -105.7540),
    ("golden_gate_sp",   "Golden Gate Canyon State Park", 39.8310, -105.4100),
    ("jackson_lake_sp",  "Jackson Lake State Park",       40.3840, -104.0910),
    ("pawnee_buttes",    "Pawnee Buttes",                 40.8220, -103.9890),
    ("denver_downtown",  "Denver (downtown)",             39.7392, -104.9903),
    ("crestone",         "Crestone",                      37.9964, -105.6997),
]

WEATHER_VARS = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                "relative_humidity_2m", "temperature_2m", "dew_point_2m", "precipitation",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]

OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
AQS = "https://aqs.epa.gov/aqsweb/airdata/daily_{param}_{year}.zip"
AQS_PARAMS = ["88101", "88502"]   # PM2.5 FRM/FEM and non-FRM/FEM
HMS = ("https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/"
       "{y}/{m}/hms_smoke{y}{m}{d}.zip")

ROOT = Path(__file__).resolve().parent / "data"
RAW, OUT = ROOT / "raw", ROOT / "processed"
MANIFEST = RAW / "manifest.csv"
UA = {"User-Agent": "CSCI5502-sky-clarity-student-project"}


# ----------------------------------------------------------------------------- helpers
def sites_df():
    return pd.DataFrame(SITES, columns=["site_id", "site_name", "latitude", "longitude"])


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def fetch(session, url, path, source, params=None, tries=4):
    """Download url to path unless cached. Returns True if file exists afterwards, False on 404."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, headers=UA, timeout=120)
        except requests.RequestException as e:
            print(f"   network error ({e}); retrying"); time.sleep(10 * (attempt + 1)); continue
        if r.status_code == 404:
            return False
        if r.status_code == 429 or r.status_code >= 500:
            wait = 65 if r.status_code == 429 else 15 * (attempt + 1)
            print(f"   HTTP {r.status_code}; waiting {wait}s"); time.sleep(wait); continue
        r.raise_for_status()
        path.write_bytes(r.content)
        new = not MANIFEST.exists()
        with open(MANIFEST, "a") as f:
            if new:
                f.write("source,file,url,accessed_utc,bytes,sha256\n")
            f.write(f'{source},{path.relative_to(RAW)},"{r.url}",'
                    f"{datetime.now(timezone.utc).isoformat(timespec='seconds')},"
                    f"{len(r.content)},{hashlib.sha256(r.content).hexdigest()}\n")
        return True
    raise RuntimeError(f"Gave up on {url}")


# ----------------------------------------------------------------------------- stage 1: download
def download(start, end):
    RAW.mkdir(parents=True, exist_ok=True)
    sites_df().to_csv(RAW / "sites.csv", index=False)
    s = requests.Session()
    years = range(int(start[:4]), int(end[:4]) + 1)

    print("[1/3] Open-Meteo historical weather (hourly, UTC)")
    # one extra day so the last night's morning hours are covered
    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for sid, _, lat, lon in SITES:
        for y in years:
            a = max(start, f"{y}-01-01")
            b = end_plus if y == years[-1] else f"{y}-12-31"
            p = RAW / "open_meteo" / f"{sid}_{y}.json"
            cached = p.exists()
            fetch(s, OPEN_METEO, p, "open_meteo",
                  params={"latitude": lat, "longitude": lon, "start_date": a, "end_date": b,
                          "hourly": ",".join(WEATHER_VARS), "timezone": "GMT",
                          "wind_speed_unit": "ms"})
            if not cached:
                time.sleep(5)      # stay under the free-tier per-minute limit
        print("   ", sid)

    print("[2/3] EPA AQS daily PM2.5 (national yearly files)")
    for y in years:
        for prm in AQS_PARAMS:
            ok = fetch(s, AQS.format(param=prm, year=y), RAW / "aqs" / f"daily_{prm}_{y}.zip", "epa_aqs")
            print("   ", prm, y, "ok" if ok else "NOT FOUND")

    print("[3/3] NOAA HMS smoke polygons (one small zip per day; takes a while)")
    miss_file = RAW / "hms" / "_missing_dates.txt"
    miss_file.parent.mkdir(parents=True, exist_ok=True)
    known_missing = set(miss_file.read_text().split()) if miss_file.exists() else set()
    n_ok = 0
    for day in pd.date_range(start, end, freq="D"):
        ymd = day.strftime("%Y%m%d")
        if ymd in known_missing:
            continue
        p = RAW / "hms" / ymd[:4] / f"hms_smoke{ymd}.zip"
        cached = p.exists()
        ok = fetch(s, HMS.format(y=ymd[:4], m=ymd[4:6], d=ymd[6:]), p, "noaa_hms")
        if ok:
            n_ok += 1
        else:
            known_missing.add(ymd)
            with open(miss_file, "a") as f:
                f.write(ymd + "\n")
        if not cached:
            time.sleep(0.2)
        if day.day == 1:
            print("   ", day.strftime("%Y-%m"))
    print(f"    {n_ok} daily files present, {len(known_missing)} dates missing at source")
    if n_ok == 0:
        print("    WARNING: no HMS files downloaded - check the HMS URL pattern at the top of this script.")

    (RAW / "globe_at_night").mkdir(exist_ok=True)
    (RAW / "viirs").mkdir(exist_ok=True)
    print("Done. Manual step: put Globe at Night yearly CSVs in", RAW / "globe_at_night")


# ----------------------------------------------------------------------------- stage 2: combine
def load_weather():
    frames, elev = [], {}
    for sid, *_ in SITES:
        for p in sorted((RAW / "open_meteo").glob(f"{sid}_*.json")):
            j = json.loads(p.read_text())
            df = pd.DataFrame(j["hourly"])
            df["time_utc"] = pd.to_datetime(df.pop("time"), utc=True)
            df["site_id"] = sid
            elev[sid] = j.get("elevation")
            frames.append(df)
    if not frames:
        sys.exit("No Open-Meteo files found - run the download stage first.")
    w = pd.concat(frames, ignore_index=True).drop_duplicates(["site_id", "time_utc"])
    return w, elev


def add_astronomy(w, elev):
    """Sun/moon altitude and moon illuminated fraction for every site-hour (skyfield, JPL DE421)."""
    from skyfield.api import Loader, wgs84
    load = Loader(str(RAW / "ephemeris"))
    (RAW / "ephemeris").mkdir(parents=True, exist_ok=True)
    eph, ts = load("de421.bsp"), load.timescale()
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    parts = []
    for sid, _, lat, lon in SITES:
        g = w[w.site_id == sid].copy()
        if g.empty:
            continue
        i = pd.DatetimeIndex(g.time_utc)
        t = ts.utc(i.year.values, i.month.values, i.day.values, i.hour.values)
        obs = (earth + wgs84.latlon(lat, lon, elevation_m=elev.get(sid) or 0)).at(t)
        g["sun_alt_deg"] = obs.observe(sun).apparent().altaz()[0].degrees
        m = obs.observe(moon).apparent()
        g["moon_alt_deg"] = m.altaz()[0].degrees
        g["moon_illum_frac"] = m.fraction_illuminated(sun)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def nightly(w):
    """Aggregate dark hours to one row per site-night. night_date = local date the evening began."""
    local = w.time_utc.dt.tz_convert(LOCAL_TZ)
    w = w.assign(night_date=(local - pd.Timedelta(hours=12)).dt.tz_localize(None).dt.normalize())
    d = w[w.sun_alt_deg < DARK_SUN_ALT].copy()
    rad = np.radians(d.wind_direction_10m)
    d["_u"], d["_v"] = np.sin(rad), np.cos(rad)
    d["_moon_up"] = (d.moon_alt_deg > 0).astype(float)
    g = d.groupby(["site_id", "night_date"])
    n = g.agg(
        n_dark_hours=("time_utc", "size"),
        cloud_cover_mean=("cloud_cover", "mean"),
        cloud_cover_max=("cloud_cover", "max"),
        cloud_low_mean=("cloud_cover_low", "mean"),
        cloud_mid_mean=("cloud_cover_mid", "mean"),
        cloud_high_mean=("cloud_cover_high", "mean"),
        rel_humidity_mean=("relative_humidity_2m", "mean"),
        temp_c_mean=("temperature_2m", "mean"),
        dew_point_c_mean=("dew_point_2m", "mean"),
        precip_mm_sum=("precipitation", "sum"),
        wind_speed_ms_mean=("wind_speed_10m", "mean"),
        wind_gust_ms_max=("wind_gusts_10m", "max"),
        _u=("_u", "mean"), _v=("_v", "mean"),
        moon_illum_frac=("moon_illum_frac", "mean"),
        moon_up_frac=("_moon_up", "mean"),
        moon_alt_max_deg=("moon_alt_deg", "max"),
    ).reset_index()
    n["wind_dir_deg_mean"] = (np.degrees(np.arctan2(n.pop("_u"), n.pop("_v"))) + 360) % 360
    return n


def load_pm25():
    """Colorado monitor-day PM2.5 means from the AQS yearly files."""
    cols = ["State Code", "County Code", "Site Num", "Latitude", "Longitude",
            "Date Local", "Event Type", "Arithmetic Mean"]
    frames = []
    for p in sorted((RAW / "aqs").glob("daily_*.zip")):
        with zipfile.ZipFile(p) as z:
            df = pd.read_csv(z.open(z.namelist()[0]), usecols=cols, low_memory=False,
                             dtype={"State Code": str, "County Code": str, "Site Num": str})
        df = df[df["State Code"].str.zfill(2) == "08"]
        # keep wildfire-affected values: drop only the "Excluded" duplicates of flagged days
        df = df[df["Event Type"].fillna("None").isin(["None", "Included"])]
        frames.append(df)
    if not frames:
        return None
    a = pd.concat(frames, ignore_index=True)
    a["monitor_id"] = ("08-" + a["County Code"].str.zfill(3) + "-" + a["Site Num"].str.zfill(4))
    a["date"] = pd.to_datetime(a["Date Local"])
    return (a.groupby(["monitor_id", "date"])
             .agg(pm25=("Arithmetic Mean", "mean"), lat=("Latitude", "first"), lon=("Longitude", "first"))
             .reset_index())


def site_pm25(pm):
    """For each site-day, the nearest monitor (within PM25_MAX_KM) that reported that day."""
    mons = pm.groupby("monitor_id")[["lat", "lon"]].first().reset_index()
    pairs = sites_df().merge(mons, how="cross")
    pairs["km"] = haversine_km(pairs.latitude, pairs.longitude, pairs.lat, pairs.lon)
    pairs = pairs[pairs.km <= PM25_MAX_KM][["site_id", "monitor_id", "km"]]
    x = pairs.merge(pm[["monitor_id", "date", "pm25"]], on="monitor_id").sort_values("km")
    x = x.groupby(["site_id", "date"], as_index=False).first()
    return x.rename(columns={"pm25": "pm25_ugm3", "monitor_id": "pm25_monitor_id", "km": "pm25_monitor_km"})


def _density_rank(v):
    """HMS density -> 1 light, 2 medium, 3 heavy (text since 2022, numeric codes before)."""
    try:
        f = float(v)
        return 1 if f < 10 else 2 if f < 20 else 3
    except (TypeError, ValueError):
        return {"light": 1, "medium": 2, "heavy": 3}.get(str(v).strip().lower(), 1)


def site_smoke():
    """Max HMS smoke density over each site for each calendar day. Cached to processed/."""
    import geopandas as gpd
    cache = OUT / "_hms_site_daily.csv"
    s = sites_df()
    if cache.exists():
        c = pd.read_csv(cache, parse_dates=["date"])
        if set(s.site_id) <= set(c.site_id):
            return c[c.site_id.isin(s.site_id)]
        print("    site list changed - recomputing smoke for all sites")
    pts = gpd.GeoDataFrame(s[["site_id"]], geometry=gpd.points_from_xy(s.longitude, s.latitude), crs=4326)
    rows, bad = [], 0
    files = sorted((RAW / "hms").glob("*/hms_smoke*.zip"))
    for k, p in enumerate(files):
        date = pd.Timestamp(re.search(r"(\d{8})", p.name).group(1))
        level = pd.Series(0, index=s.site_id)
        try:
            g = gpd.read_file(f"zip://{p}")
            if len(g):
                g = g[g.geometry.notna()]
                g = (g.set_crs(4326) if g.crs is None else g.to_crs(4326))
                g["geometry"] = g.geometry.buffer(0)
                dcol = next((c for c in g.columns if c.lower() == "density"), None)
                g["_r"] = g[dcol].map(_density_rank) if dcol else 1
                hit = gpd.sjoin(pts, g[["_r", "geometry"]], predicate="within")
                level.update(hit.groupby("site_id")["_r"].max())
        except Exception:
            bad += 1
            level[:] = np.nan
        rows.append(pd.DataFrame({"site_id": level.index, "date": date, "hms_smoke_level": level.values}))
        if k % 250 == 0:
            print(f"    HMS {k}/{len(files)}")
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(cache, index=False)
    print(f"    HMS: {len(files)} daily files read, {bad} unreadable")
    return out


def site_viirs():
    """Optional: sample EOG annual VIIRS radiance (nW/cm2/sr) at each site from GeoTIFFs."""
    # Accepts .tif or the .tif.gz exactly as downloaded. A .gz is unpacked to a temporary file,
    # sampled, then the temporary file is deleted (needs ~12 GB free while it runs).
    # Results are cached, so each file is only processed once.
    import gzip, shutil
    cache = OUT / "_viirs_site_year.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["site_id", "year", "viirs_radiance"])
    files = sorted(list((RAW / "viirs").glob("*.tif")) + list((RAW / "viirs").glob("*.tif.gz")))
    if not files and done.empty:
        return None
    import rasterio
    s = sites_df()
    for p in files:
        m = re.search(r"_(20\d{2})_", p.name)
        if not m:
            print("    skipping VIIRS file without _YYYY_ in name:", p.name); continue
        year = int(m.group(1))
        if (done.year == year).sum() >= len(s):
            continue
        tmp = None
        try:
            if p.suffix == ".gz":
                print(f"    VIIRS {year}: unpacking {p.name} (several minutes)")
                tmp = p.with_name("_tmp_unpacked.tif")
                with gzip.open(p, "rb") as fi, open(tmp, "wb") as fo:
                    shutil.copyfileobj(fi, fo, length=16 * 1024 * 1024)
            with rasterio.open(tmp or p) as src:
                vals = [float(v[0]) for v in src.sample(zip(s.longitude, s.latitude))]
        except Exception as e:
            print(f"    VIIRS {year}: could not read {p.name} ({e}) - is the download complete?"); continue
        finally:
            if tmp and tmp.exists():
                tmp.unlink()
        new = pd.DataFrame({"site_id": s.site_id, "year": year, "viirs_radiance": vals})
        done = pd.concat([done[done.year != year], new], ignore_index=True)
        done.to_csv(cache, index=False)
    return done.astype({"year": int}) if len(done) else None


def load_gan():
    files = sorted((RAW / "globe_at_night").glob("*.csv"))
    if not files:
        return None
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip", encoding="latin-1")
        df.columns = [re.sub(r"[^a-z0-9]", "", c.lower()) for c in df.columns]
        df["gan_source_file"] = p.name
        frames.append(df)
    g = pd.concat(frames, ignore_index=True)
    need = ["latitude", "longitude", "localdate", "localtime", "limitingmag"]
    missing = [c for c in need if c not in g.columns]
    if missing:
        print("    Globe at Night: expected columns not found:", missing, "\n    columns seen:", list(g.columns))
        return None
    for c in ["latitude", "longitude", "limitingmag", "sqmreading", "cloudcover", "elevationm"]:
        if c in g.columns and c != "cloudcover":
            g[c] = pd.to_numeric(g[c], errors="coerce")
    la0, la1, lo0, lo1 = GAN_BBOX
    g = g[g.latitude.between(la0, la1) & g.longitude.between(lo0, lo1)].copy()
    g.loc[g.limitingmag < 0, "limitingmag"] = np.nan
    if "sqmreading" in g.columns:
        g.loc[g.sqmreading <= 0, "sqmreading"] = np.nan
    ts = pd.to_datetime(g.localdate.astype(str) + " " + g.localtime.astype(str), errors="coerce")
    g["obs_local_time"] = ts
    g["night_date"] = (ts - pd.Timedelta(hours=12)).dt.normalize()
    keep = ["gan_source_file", "obs_local_time", "night_date", "latitude", "longitude", "limitingmag"] + \
           [c for c in ["id", "obsid", "elevationm", "sqmreading", "cloudcover", "constellation"] if c in g.columns]
    g = g[keep].rename(columns={"latitude": "obs_latitude", "longitude": "obs_longitude",
                                "limitingmag": "limiting_mag", "sqmreading": "sqm_reading",
                                "cloudcover": "observer_cloud_cover", "elevationm": "obs_elevation_m",
                                "id": "gan_id", "obsid": "gan_obs_id"})
    return g.reset_index(drop=True)


def combine():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = [f"Build report  {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    s = sites_df()

    print("Weather + astronomy")
    w, elev = load_weather()
    d1 = nightly(add_astronomy(w, elev))
    d1 = s.assign(elevation_m=s.site_id.map(elev)).merge(d1, on="site_id")

    print("PM2.5")
    pm = load_pm25()
    if pm is not None:
        sp = site_pm25(pm)
        d1 = d1.merge(sp.rename(columns={"date": "night_date"}), on=["site_id", "night_date"], how="left")
        nxt = sp[["site_id", "date", "pm25_ugm3"]].rename(columns={"pm25_ugm3": "pm25_next_day_ugm3"})
        nxt["night_date"] = nxt.pop("date") - pd.Timedelta(days=1)
        d1 = d1.merge(nxt, on=["site_id", "night_date"], how="left")
    else:
        rep.append("PM2.5: no AQS files found")

    print("HMS smoke")
    sm = site_smoke()
    if sm is not None:
        d1 = d1.merge(sm.rename(columns={"date": "night_date"}), on=["site_id", "night_date"], how="left")
    else:
        rep.append("HMS: no smoke files found")

    v = site_viirs()
    if v is not None:
        d1["year"] = d1.night_date.dt.year
        d1 = d1.merge(v, on=["site_id", "year"], how="left").drop(columns="year")
    else:
        rep.append("VIIRS: no GeoTIFFs in data/raw/viirs - viirs_radiance column omitted")

    d1 = d1.sort_values(["site_id", "night_date"]).reset_index(drop=True)
    assert not d1.duplicated(["site_id", "night_date"]).any(), "duplicate site-night keys"
    d1.to_csv(OUT / "dataset1_site_night.csv", index=False, date_format="%Y-%m-%d")
    rep += [f"Dataset 1: {len(d1):,} rows, {d1.site_id.nunique()} sites, "
            f"{d1.night_date.min().date()} to {d1.night_date.max().date()}",
            "Missing values per column (%):",
            (d1.isna().mean() * 100).round(2).to_string(), ""]

    print("Globe at Night")
    g = load_gan()
    if g is None:
        rep.append("Dataset 2: not built (no usable CSVs in data/raw/globe_at_night)")
    else:
        dist = haversine_km(g.obs_latitude.values[:, None], g.obs_longitude.values[:, None],
                            s.latitude.values[None, :], s.longitude.values[None, :])
        j = dist.argmin(axis=1)
        g["site_id"] = s.site_id.values[j]
        g["site_km"] = dist[np.arange(len(g)), j]
        g["within_radius"] = g.site_km <= GAN_MAX_KM
        d2 = g.merge(d1.drop(columns=["site_name", "latitude", "longitude"]),
                     on=["site_id", "night_date"], how="left", indicator=True)
        d2["matched"] = d2.within_radius & (d2.pop("_merge") == "both")
        d1_cols = [c for c in d1.columns if c not in ("site_id", "night_date", "site_name", "latitude", "longitude")]
        d2.loc[~d2.matched, d1_cols] = np.nan     # too far from any site: keep the row, blank the conditions
        d2.to_csv(OUT / "dataset2_gan_observations.csv", index=False)
        rep += [f"Dataset 2: {len(d2):,} Globe at Night observations in Colorado bounding box",
                f"  within {GAN_MAX_KM} km of a site: {int(g.within_radius.sum()):,}",
                f"  matched to a dataset-1 site-night: {int(d2.matched.sum()):,}",
                f"  unmatched (too far, bad timestamp, or outside date range): {int((~d2.matched).sum()):,}",
                f"  matched with a limiting magnitude: {int((d2.matched & d2.limiting_mag.notna()).sum()):,}",
                "  matched observations per site:",
                d2[d2.matched].site_id.value_counts().to_string()]

    (OUT / "build_report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "combine", "all"])
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    a = ap.parse_args()
    if a.stage in ("download", "all"):
        download(a.start, a.end)
    if a.stage in ("combine", "all"):
        combine()
