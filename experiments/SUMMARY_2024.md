# Gate 1 reference scorecards, all of 2024: summary

Climatology, persistence and raw NOAA GEFS on the 2024 validation split, 00Z and 12Z runs, 500 stations,
one scorecard per model and month (36 files in this folder). Every number below is read from those
scorecards; nothing was recomputed from data. The tables are printed by

```sh
uv run python scripts/run_gate1.py summary --scorecards experiments/ --year 2024
```

and a test checks that this file holds exactly that output.

## How to read the tables

- **Skill** is `crpss_vs_baseline`: one minus the model's composite loss over climatology's, in percent,
  on the same points. Positive means better than climatology. "fair" uses the fair CRPS (the headline),
  "ecdf" the plain ensemble CRPS.
- Every number is the scorecard's `"all"` entry, the average over all leads **that model has**: 3 to 48 h
  for GEFS, 1 to 48 h for persistence. So the GEFS and persistence columns of one row are not on the same
  leads and should not be subtracted from each other. Per-lead values are in the scorecards.
- **Spread-skill** is ensemble spread over the error of the ensemble mean; 1 is calibrated, below 1 is
  overconfident (too little spread). The gate accepts 0.85 to 1.15.
- **Gate** lists the gates that variable fails in that month: `skill` (skill must be above 5 %), `tail`
  (tail loss not worse than climatology's), `spread_skill` (band above), `dm` (Diebold-Mariano p below
  0.05), `min_members` (at least 20 members). A scorecard passes only if all five variables pass.
- **December is 17 days.** It stops at the init of 2024-12-17T12Z, the last one inside the validation split
  (`configs/splits.yaml` ends it on 2024-12-17T18Z, 14 days before the sealed 2025 split). All three models
  share that coverage: 34 init times instead of 62. The year has 704 init times in all.

## Reading

**No model passes the gate in any month.** All 36 scorecards have `passed_gate` false. Climatology
cannot pass by construction (it is the baseline).

**Raw GEFS beats climatology in most of the year, by a lot in the cold season, and loses or ties in high
summer for moisture and the east-west wind.**

- Over all leads GEFS scores below climatology in three variable-months: `td2m` in July (-2.4 %), `u10`
  in July (-4.1 %) and `u10` in October (-1.0 %).
- It is positive but under the 5 % bar in three more: `t2m` in July (+3.8 %), `td2m` in August (+1.4 %)
  and `u10` in August (+1.9 %). Those six are the six `skill` gate failures of GEFS in the year.
- At single leads GEFS is below climatology 17 times, all in July, August and October (table at the end):
  `td2m` at every lead in July and at 6 h in August, `u10` from 6 h on in July and from 12 h on in August
  and October, and `t2m` at 3 h in July.
- The seasonal swing is large. `t2m` skill falls from +56 % in January to +4 % in July and is back at
  +45 % in November. `td2m` goes from +66 % to -2 % and back to +57 %. `u10` is between +14 % and +29 %
  from November to June and between -4 % and +6 % from July to October. `v10` stays positive all year
  (+10 % to +31 %). `msl` is the strong variable: +55 % (July) to +81 % (February).
- The plain-ensemble (ecdf) skill of GEFS is 0.3 to 3.1 points higher than the fair skill in every
  variable-month, which is enough to flip the sign in the two closest cases (`td2m` July +0.2 %, `u10`
  October +2.1 %). `u10` in July is negative either way (-1.8 %).
- The scorecards say how much, not why. Two things are known from other work in this repository and are
  consistent with the pattern, but are not shown by these numbers: raw GEFS grid values are compared with
  point observations without any bias correction, and `docs/BASELINES.md` documents terrain and nighttime
  temperature biases.

**Raw GEFS is badly overconfident all year.** Its spread-skill ratio is between 0.23 and 0.58 in all 60
variable-months and never inside the 0.85 to 1.15 band, so `spread_skill` fails 60 times out of 60 and
is the only gate GEFS fails in 36 of them. The spread is between a quarter and a little over half of the
error it should match.

- `t2m` is the worst: 0.23 to 0.38, lowest from October to February (0.23 to 0.27) and highest from May
  to August (0.35 to 0.38).
- `td2m`: 0.34 to 0.47, highest in spring (0.44 to 0.47 from March to May), lowest from July to October
  (0.34 to 0.35).
- Winds: 0.35 to 0.49, highest from May to September, lowest in winter.
- `msl`: 0.39 to 0.58, highest from March to June (0.54 to 0.58), lowest in December and January
  (0.40, 0.39).
- So calibration is somewhat better in the warm half of the year for temperature and wind, exactly when
  skill is lowest. It never comes close to the band. Part of this is expected for a raw model ensemble
  scored against stations: it carries no spread for the difference between a grid cell and a point.
- GEFS also fails the `tail` gate in 23 variable-months (mostly winds, and `t2m` from July to
  September) and `dm` in 5 (the months where its skill is near zero).

**Persistence** (the last observation, with spread from past changes) is not a like-for-like comparison
with GEFS here because its average includes the 1 h lead.

- Over its leads it beats climatology for `t2m` (+3 % in July to +36 % in January), `td2m` (+24 % to
  +53 %) and `msl` (+40 % to +53 %). For both wind components it is at or below climatology in every
  month (-20 % to +3 %): it wins at 1 h in every month, at 3 h in every month but one (`u10` in July),
  and loses at 12 h and beyond in every month.
- Its calibration swings with the season: inside the band in the cold months for most variables,
  over-dispersed in summer (`t2m` 1.20 to 1.32 from May to August, `td2m` up to 1.48 and `msl` up to
  1.89 in July) and under-dispersed for `msl` in December and January (0.79, 0.75).
- Five variable-months pass all gates (`td2m` in January, February and November, `msl` in April and
  October), never all five variables of a month. It fails `tail` in 51 of 60.

**Climatology** is the calibrated one: its spread-skill ratio is inside the band in 59 of 60
variable-months (0.88 to 1.14; the exception is `t2m` in May at 1.19).

## Tables

<!-- tables:start -->
### `t2m`

| Month | Init times | GEFS skill fair % | GEFS skill ecdf % | GEFS spread-skill | GEFS gate | Persistence skill fair % | Persistence skill ecdf % | Persistence spread-skill | Persistence gate | Climatology spread-skill |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- | ---: |
| 2024-01 | 62 | +56.0 | +57.1 | 0.27 | fails spread_skill | +36.4 | +36.4 | 0.89 | fails tail | 0.89 |
| 2024-02 | 58 | +51.4 | +52.8 | 0.27 | fails spread_skill | +31.4 | +31.7 | 0.87 | fails tail | 1.05 |
| 2024-03 | 62 | +45.1 | +46.7 | 0.33 | fails spread_skill | +23.2 | +23.4 | 0.96 | fails tail | 1.07 |
| 2024-04 | 60 | +36.8 | +38.6 | 0.33 | fails spread_skill | +14.7 | +15.0 | 0.98 | fails tail | 1.08 |
| 2024-05 | 62 | +24.4 | +26.7 | 0.36 | fails spread_skill | +12.6 | +12.5 | 1.20 | fails tail, spread_skill | 1.19 |
| 2024-06 | 60 | +12.8 | +15.1 | 0.37 | fails spread_skill | +5.8 | +5.2 | 1.22 | fails tail, spread_skill, dm | 1.05 |
| 2024-07 | 62 | +3.8 | +6.2 | 0.35 | fails skill, tail, spread_skill | +3.3 | +2.1 | 1.32 | fails skill, tail, spread_skill | 0.98 |
| 2024-08 | 62 | +15.3 | +17.2 | 0.38 | fails tail, spread_skill | +6.1 | +5.0 | 1.27 | fails tail, spread_skill, dm | 0.94 |
| 2024-09 | 60 | +21.8 | +23.9 | 0.33 | fails tail, spread_skill | +6.5 | +6.3 | 1.14 | fails tail | 1.07 |
| 2024-10 | 62 | +33.3 | +35.1 | 0.26 | fails spread_skill | +18.5 | +18.5 | 0.99 | fails tail | 1.01 |
| 2024-11 | 60 | +44.9 | +46.5 | 0.26 | fails spread_skill | +25.8 | +25.9 | 1.00 | fails tail | 1.07 |
| 2024-12 | 34 | +36.5 | +38.3 | 0.23 | fails spread_skill | +19.2 | +19.4 | 0.89 | fails tail | 1.01 |

### `td2m`

| Month | Init times | GEFS skill fair % | GEFS skill ecdf % | GEFS spread-skill | GEFS gate | Persistence skill fair % | Persistence skill ecdf % | Persistence spread-skill | Persistence gate | Climatology spread-skill |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- | ---: |
| 2024-01 | 62 | +66.1 | +66.8 | 0.36 | fails spread_skill | +52.7 | +52.6 | 0.87 | pass | 0.88 |
| 2024-02 | 58 | +58.5 | +59.6 | 0.40 | fails spread_skill | +43.4 | +43.6 | 0.87 | pass | 1.06 |
| 2024-03 | 62 | +53.0 | +54.2 | 0.44 | fails spread_skill | +35.7 | +35.9 | 0.91 | fails tail | 1.06 |
| 2024-04 | 60 | +46.1 | +47.5 | 0.44 | fails spread_skill | +32.4 | +32.5 | 0.98 | fails tail | 1.09 |
| 2024-05 | 62 | +40.5 | +42.0 | 0.47 | fails spread_skill | +31.5 | +31.3 | 1.17 | fails tail, spread_skill | 1.10 |
| 2024-06 | 60 | +28.1 | +29.8 | 0.43 | fails spread_skill | +24.7 | +24.1 | 1.18 | fails tail, spread_skill | 0.99 |
| 2024-07 | 62 | -2.4 | +0.2 | 0.34 | fails skill, tail, spread_skill, dm | +23.9 | +22.6 | 1.48 | fails tail, spread_skill | 1.00 |
| 2024-08 | 62 | +1.4 | +3.7 | 0.35 | fails skill, spread_skill, dm | +30.5 | +29.3 | 1.46 | fails tail, spread_skill | 0.96 |
| 2024-09 | 60 | +12.2 | +14.6 | 0.34 | fails spread_skill | +34.7 | +34.1 | 1.34 | fails tail, spread_skill | 1.09 |
| 2024-10 | 62 | +28.3 | +30.5 | 0.34 | fails spread_skill | +37.5 | +37.5 | 1.13 | fails tail | 1.14 |
| 2024-11 | 60 | +55.4 | +56.4 | 0.38 | fails spread_skill | +41.4 | +41.4 | 0.95 | pass | 0.98 |
| 2024-12 | 34 | +57.5 | +58.5 | 0.39 | fails spread_skill | +37.3 | +37.4 | 0.80 | fails tail, spread_skill | 0.92 |

### `u10`

| Month | Init times | GEFS skill fair % | GEFS skill ecdf % | GEFS spread-skill | GEFS gate | Persistence skill fair % | Persistence skill ecdf % | Persistence spread-skill | Persistence gate | Climatology spread-skill |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- | ---: |
| 2024-01 | 62 | +23.0 | +24.7 | 0.36 | fails spread_skill | +0.4 | +0.2 | 0.94 | fails skill, tail, dm | 0.95 |
| 2024-02 | 58 | +20.2 | +22.3 | 0.38 | fails tail, spread_skill | -0.2 | -0.2 | 0.98 | fails skill, tail, dm | 1.04 |
| 2024-03 | 62 | +23.9 | +25.7 | 0.41 | fails tail, spread_skill | +0.3 | +0.5 | 0.90 | fails skill, tail, dm | 0.99 |
| 2024-04 | 60 | +29.3 | +30.9 | 0.43 | fails spread_skill | -0.4 | -0.0 | 0.83 | fails skill, tail, spread_skill, dm | 0.95 |
| 2024-05 | 62 | +19.4 | +21.1 | 0.49 | fails spread_skill | -10.4 | -10.1 | 0.87 | fails skill, tail, dm | 0.96 |
| 2024-06 | 60 | +13.9 | +15.6 | 0.48 | fails tail, spread_skill | -10.8 | -10.9 | 0.95 | fails skill, tail, dm | 0.96 |
| 2024-07 | 62 | -4.1 | -1.8 | 0.48 | fails skill, tail, spread_skill, dm | -19.6 | -20.0 | 1.09 | fails skill, tail, dm | 1.02 |
| 2024-08 | 62 | +1.9 | +4.0 | 0.47 | fails skill, tail, spread_skill, dm | -12.4 | -13.2 | 1.10 | fails skill, tail, dm | 0.96 |
| 2024-09 | 60 | +5.5 | +7.6 | 0.45 | fails spread_skill | -8.0 | -8.8 | 1.17 | fails skill, tail, spread_skill, dm | 1.00 |
| 2024-10 | 62 | -1.0 | +2.1 | 0.37 | fails skill, tail, spread_skill, dm | -8.7 | -9.0 | 1.19 | fails skill, tail, spread_skill, dm | 1.14 |
| 2024-11 | 60 | +22.1 | +23.8 | 0.38 | fails tail, spread_skill | +2.5 | +2.0 | 1.00 | fails skill, tail, dm | 0.93 |
| 2024-12 | 34 | +15.4 | +17.5 | 0.35 | fails tail, spread_skill | +0.9 | +0.6 | 1.05 | fails skill, tail, dm | 0.99 |

### `v10`

| Month | Init times | GEFS skill fair % | GEFS skill ecdf % | GEFS spread-skill | GEFS gate | Persistence skill fair % | Persistence skill ecdf % | Persistence spread-skill | Persistence gate | Climatology spread-skill |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- | ---: |
| 2024-01 | 62 | +24.9 | +27.0 | 0.37 | fails tail, spread_skill | +1.1 | +1.1 | 1.04 | fails skill, tail, dm | 1.07 |
| 2024-02 | 58 | +30.0 | +31.8 | 0.39 | fails tail, spread_skill | +2.2 | +2.3 | 0.96 | fails skill, tail, dm | 1.05 |
| 2024-03 | 62 | +30.7 | +32.4 | 0.42 | fails tail, spread_skill | +1.4 | +1.7 | 0.91 | fails skill, tail, dm | 1.03 |
| 2024-04 | 60 | +29.9 | +31.6 | 0.43 | fails tail, spread_skill | -1.4 | -0.9 | 0.89 | fails skill, tail, dm | 1.04 |
| 2024-05 | 62 | +19.1 | +21.0 | 0.48 | fails tail, spread_skill | -10.4 | -10.1 | 0.95 | fails skill, tail, dm | 1.06 |
| 2024-06 | 60 | +18.7 | +20.5 | 0.48 | fails tail, spread_skill | -12.0 | -12.0 | 0.97 | fails skill, tail, dm | 1.00 |
| 2024-07 | 62 | +9.5 | +11.5 | 0.48 | fails tail, spread_skill | -10.6 | -11.2 | 1.12 | fails skill, tail, dm | 0.99 |
| 2024-08 | 62 | +10.2 | +12.1 | 0.47 | fails tail, spread_skill | -9.7 | -10.4 | 1.14 | fails skill, tail, dm | 0.98 |
| 2024-09 | 60 | +17.1 | +19.0 | 0.45 | fails spread_skill | -2.0 | -2.7 | 1.19 | fails skill, tail, spread_skill, dm | 1.01 |
| 2024-10 | 62 | +20.0 | +22.1 | 0.38 | fails tail, spread_skill | +2.4 | +2.0 | 1.12 | fails skill, tail, dm | 1.02 |
| 2024-11 | 60 | +20.5 | +22.6 | 0.39 | fails tail, spread_skill | +0.9 | +0.7 | 1.07 | fails skill, tail, dm | 1.05 |
| 2024-12 | 34 | +18.2 | +20.5 | 0.37 | fails tail, spread_skill | -2.1 | -2.3 | 1.10 | fails skill, tail, dm | 1.07 |

### `msl`

| Month | Init times | GEFS skill fair % | GEFS skill ecdf % | GEFS spread-skill | GEFS gate | Persistence skill fair % | Persistence skill ecdf % | Persistence spread-skill | Persistence gate | Climatology spread-skill |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- | ---: |
| 2024-01 | 62 | +76.9 | +77.4 | 0.39 | fails spread_skill | +41.5 | +42.0 | 0.75 | fails spread_skill | 0.98 |
| 2024-02 | 58 | +80.9 | +81.3 | 0.49 | fails spread_skill | +44.2 | +44.6 | 0.82 | fails tail, spread_skill | 1.02 |
| 2024-03 | 62 | +78.4 | +78.8 | 0.54 | fails spread_skill | +44.4 | +44.6 | 0.92 | fails tail | 1.00 |
| 2024-04 | 60 | +76.7 | +77.2 | 0.57 | fails spread_skill | +41.3 | +41.4 | 0.97 | pass | 1.02 |
| 2024-05 | 62 | +75.1 | +75.4 | 0.58 | fails spread_skill | +51.8 | +51.2 | 1.27 | fails tail, spread_skill | 0.90 |
| 2024-06 | 60 | +69.4 | +69.9 | 0.54 | fails spread_skill | +42.4 | +41.6 | 1.29 | fails spread_skill | 0.91 |
| 2024-07 | 62 | +55.4 | +56.3 | 0.48 | fails spread_skill | +39.7 | +38.1 | 1.89 | fails tail, spread_skill | 1.01 |
| 2024-08 | 62 | +58.7 | +59.4 | 0.48 | fails spread_skill | +45.1 | +43.4 | 1.87 | fails spread_skill | 0.93 |
| 2024-09 | 60 | +69.2 | +69.8 | 0.50 | fails spread_skill | +52.8 | +51.8 | 1.58 | fails tail, spread_skill | 0.96 |
| 2024-10 | 62 | +73.1 | +73.7 | 0.45 | fails spread_skill | +44.8 | +44.7 | 1.12 | pass | 1.10 |
| 2024-11 | 60 | +79.4 | +79.8 | 0.49 | fails spread_skill | +44.3 | +44.5 | 0.91 | fails tail | 1.03 |
| 2024-12 | 34 | +77.8 | +78.3 | 0.40 | fails spread_skill | +41.7 | +42.0 | 0.79 | fails spread_skill | 0.97 |

### Single leads at which GEFS scores below climatology

| Month | Variable | Lead h | GEFS skill fair % | GEFS skill ecdf % |
| :--- | :--- | ---: | ---: | ---: |
| 2024-07 | `t2m` | 3 | -1.2 | +1.5 |
| 2024-07 | `td2m` | 3 | -0.3 | +2.4 |
| 2024-07 | `td2m` | 6 | -5.6 | -2.8 |
| 2024-07 | `td2m` | 12 | -1.3 | +1.3 |
| 2024-07 | `td2m` | 24 | -1.6 | +0.9 |
| 2024-07 | `td2m` | 48 | -2.9 | -0.7 |
| 2024-07 | `u10` | 6 | -4.2 | -1.5 |
| 2024-07 | `u10` | 12 | -5.3 | -2.8 |
| 2024-07 | `u10` | 24 | -6.0 | -3.7 |
| 2024-07 | `u10` | 48 | -8.2 | -6.2 |
| 2024-08 | `td2m` | 6 | -4.2 | -1.6 |
| 2024-08 | `u10` | 12 | -1.1 | +1.1 |
| 2024-08 | `u10` | 24 | -1.3 | +0.7 |
| 2024-08 | `u10` | 48 | -1.2 | +0.5 |
| 2024-10 | `u10` | 12 | -5.8 | -2.4 |
| 2024-10 | `u10` | 24 | -4.8 | -1.7 |
| 2024-10 | `u10` | 48 | -3.3 | -0.5 |
<!-- tables:end -->
