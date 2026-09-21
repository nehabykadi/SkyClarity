# SkyClarity

**Can you see the sky tonight?** A project to predict night sky visibility in the Boulder, CO area from weather, wildfire smoke, moonlight, and light pollution data.

CSCI 5502 (Data Mining) — CU Boulder

---

## Team Members

| Name | Role |
|---|---|
| Neha Ramachandra | Coordination and Analysis / Modeling |
| Daralynn Rhode | Data and Visualization |
| Shreyas Kulkarni | Documentation and Reproducibility |

## Project Goal

Stargazing conditions in the Boulder, CO area depend on a combination of factors — light pollution, cloud cover, wildfire smoke, and moon or twilight conditions — that aren't currently combined into a single predictive tool. Stargazers, astro-tourism operators, and astronomy educators are left to piece it together themselves, or just guess.

Our goal is to build a model that predicts night sky visibility quality for a given date and time window in Boulder, producing a score or classification that lets people plan ahead rather than relying on guesswork.

**Primary research question:** Can night sky visibility quality in the Boulder, CO area be predicted from a combination of weather, air quality, light pollution, and astronomical factors on a given night?

## Current Research Questions

1. Given near-term forecast data (cloud cover, air quality forecast, moon phase), how well can next-night sky visibility conditions be estimated in advance?
2. Which factor most degrades visibility — cloud cover, smoke concentration, or moon illumination — and do these factors interact with each other?
3. How does night sky visibility differ between wildfire season and the rest of the year?
4. What does the distribution of night sky visibility conditions in Boulder look like across a multi-year period, in terms of cloud cover, smoke levels, and moon illumination?

## Data Sources

| Source | Provides |
|---|---|
| [NWS API](https://www.weather.gov/documentation/services-web-api) | Real-time cloud cover and short-term forecast conditions |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | ~5 years of hourly historical cloud cover and weather data |
| [EPA AirNow API](https://docs.airnowapi.org/) | Hourly PM2.5 and AQI readings, used as a proxy for wildfire smoke concentration |
| [VIIRS / World Atlas of Artificial Sky Brightness](https://www.lightpollutionmap.info/) | Nighttime lights satellite data providing an annual light pollution baseline |
| [U.S. Naval Observatory API](https://aa.usno.navy.mil/data/api) | Moon phase and astronomical twilight times for each night in the study window |

### References

- Falchi, F., Cinzano, P., Duriscoe, D., Kyba, C. C. M., Elvidge, C. D., Baugh, K., Portnov, B. A., Rybnikova, N. A., & Furgoni, R. (2016). The new world atlas of artificial night sky brightness. *Science Advances, 2*(6), e1600377.
- U.S. Environmental Protection Agency. (2001, November). *Visibility in mandatory federal Class I areas (1994–1998): A report to Congress* (Appendix C: Methodology for calculating light extinction from monitored aerosol mass data).
- King, B. (2025, June). Wildfires imperil our night skies. *Sky & Telescope*.

### Known Constraints

- No direct ground-truth measure of "dark sky visibility" exists, so a proxy (e.g., sky quality meter readings or citizen reports) is needed — introducing possible bias.
- Wildfire smoke is seasonal, which could skew the model toward certain times of year.
- Reliable predictions are likely limited to a roughly 24-hour window, since weather and cloud cover forecasts degrade in accuracy beyond that.
- No ability to independently record or check data on the ground, which limits confidence in the model's prediction accuracy.

## Milestone Roadmap CHANGE ME

> Draft schedule — adjust dates/scope to match the actual syllabus milestones.

| Milestone | Target | Status |
|---|---|---|
| Project proposal & data source identification | Complete | ✅ Done |
| Data collection & ingestion pipeline (5 sources) | TBD | ⬜ Not started |
| Data cleaning, alignment, and proxy-label construction | TBD | ⬜ Not started |
| Exploratory data analysis & feature engineering | TBD | ⬜ Not started |
| Baseline model / predictive scoring approach | TBD | ⬜ Not started |
| Model evaluation & refinement | TBD | ⬜ Not started |
| Final report & presentation | TBD | ⬜ Not started |

---

*SkyClarity — a CU Boulder Data Science project*
