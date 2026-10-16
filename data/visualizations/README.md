# Analysis of 6 Figures

Almost all the figures heavily rely on understanding the limiting_mag column so below is a reference.

| **Limiting magnitude** | **What you see** |
| :---               | :---         |
| 2-3 | Heavily light-polluted, only a few dozen stars |
| 4 | Suburban sky, major constellations visible |
| 5-6 | Rural/Dark-sky, Milky Way starts to show |
| 6-7 | Excellent dark sky, Thousands of stars visible |

### Figure 1: Cloud Cover vs. Limiting Magnitude
r = -0.12. More cloud goes with a lower limiting magnitude as expected but the correlation is not strong.
Cloud explains about 1.5% of the variance. 

### Figure 2: Light Pollution by site
Crestone averages 6.1(limiting_mag), while downtown Denver (2.7), and downtown Boulder (2.8).
Some places do not fall under what we expect because Sommers-Bausch is the worst site despite a lower radiance than Denver.
Golden Gate SP has a radiance of 0 but only 3.6 limiting magnitude with n=18.
Longmont with the same radiance as boulder averages 3.9, better than Boulder.
This means we are limited by low observation counts and confounding variables.

### Figure 3: Moonlight
The moon pattern is not what you’d expect: the means are roughly flat (3.3–3.7), and “moon up” is slightly higher in the 0.5–0.75 bin (3.73 vs. 3.35).
This does not mean the moon does not matter. Some bins have under 100 reports, there is no moon up data at low illumination and no moon mostly down data at high illumination so some comparisons can't be made.
To isolate the moon, we need moon illumination data with site fixed effects or comparison within a single site.

### Figure 4: Smoke
The means by smoke level are 3.33 (none), 4.45 (light), 3.83 (medium) and 2.60 (heavy), with n = 1,981, 78, 24 and 5.
Light smoke looks better than none, which is physically implausible. The likely cause is confounding: those 78 reports may come disproportionately from dark sites like Crestone or Gross Reservoir, or from summer.
The heavy-smoke mean (2.6) rests on 5 reports, so it isn’t evidence of anything.
The data can't support a smoke effect right now and we would need to explore more sources to incorporate this.

### Figure 5: Seasonality
The share of clear, moonless site-nights is highest in June–September (about 14%) and lowest in December (4%), with a smaller bump in May. Smoke presence climbs from about 11% in May to 33% in June, 52% in July and 60% in August, then falls to 39% in September and 17% in October.
Good-sky nights are rare year-round. Even in the best months, only about 1 night in 7 is clear and moonless.
Summer is the best season for darkness and also the smoke season. Smoke overlaps most with the clear, moonless nights.

### Figure 6: Is the smoke result a site and season artifact?
Neither, the smoke effect still exists after adjusting for sites. The heatmap shows that level 1 and 2 reports come mostly from Crestone Denver, Longmont, and Sommers=Bausch.
Level 3 only has 5 reports making is unreliable. Crestone's smoke reports come from 2022-2023 so there can be confounding variables from that time period.


