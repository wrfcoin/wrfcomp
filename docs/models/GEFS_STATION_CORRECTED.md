# GEFS station correction, version 1

This candidate adjusts the bias and spread of each 31-member GEFS station forecast
using recent verified forecasts. Gate 1 measured large local temperature biases and
under-dispersion in raw GEFS. The project registered one design (M = 1), with a
30-day window, a ten-pair minimum, and a spread multiplier between 1 and 6.
[Measured baseline limits](../BASELINES.md#known-limits-on-real-data-january-and-july-2024)

## Information available at issue time

For an initialization T and lead L, use earlier raw GEFS forecasts at the same UTC
issue hour, lead, station and variable. A historical issue I qualifies only when
I + L ≤ T. Its finite ISD observation must have a valid variable-specific offset in
[−15, +15] minutes and a measurement time in (T − 30 days, T]. A positive offset
at the current hour therefore excludes that observation. Missing reports are
discarded. History begins in January 2024 and continues across month boundaries.

Let e be the historical raw ensemble mean minus its paired observation. The bias
estimate b is the arithmetic mean of e. For eligible pairs, compute the spread multiplier
`s = sqrt(mean((e − b)²) / mean(ensemble variance))`.
Ensemble variance uses the 31 members with divisor 30, as in the verification
protocol. Clip s to [1, 6]. With zero historical ensemble variance use 6 for positive
residual MSE, or 1 when both are zero. For each current member x with raw ensemble
mean μ, the output is (μ − b) + s(x − μ).

Fewer than ten pairs leaves that variable's raw members unchanged before the final
memberwise dewpoint cap. Fit t2m, td2m, u10, v10 and msl independently, then take the
minimum of td2m and t2m. The output is float32 with the GEFS member ordering and its
3, 6, 12, 24 and 48 h leads. The model ID is `gefs-station-corrected`, version `1`.

## Reproduce a monthly forecast

From the checkout, with the existing 2024 ISD truth and per-cycle GEFS point stores
in the external data directory, run:

```sh
uv run --offline python scripts/build_candidate.py --period 2024-01 --period 2024-07
```

The builder reads each month's initialization schedule from its committed GEFS
reference scorecard. December ends at 2024-12-17 12Z. It writes
`$WRFCOMP_DATA_DIR/candidates/gefs-station-corrected/YYYY-MM.zarr` (default data
directory `~/wrfcomp-data`), reopens each store, and prints the harness content hash
and fallback counts. A rerun compares the rebuilt forecast to an existing store.

| Period | Raw fallback points / total | Forecast content SHA-256 |
| --- | ---: | --- |
| 2024-01 | 259,570 / 775,000 | `5023a440639ec5726da2d6397613bfc1932f8b6a645abc9173e39ed0a9472e42` |
| 2024-07 | 20 / 775,000 | `7b527327670a22f66029a8046dfe5eed0d9a338ae0ea462ee40d335ce9058c18` |

One point is one variable, issue, lead and station, counted once for all 31 members.
January fallback counts were 51,513 for t2m, 51,585 for td2m, 52,260 each for u10
and v10, and 51,952 for msl. July had ten each for t2m and td2m.

## Preliminary, not a promotion verdict

All-lead fair composite skill over each reference, and the candidate spread-skill
ratio, from four local wrfcomp-eval runs. Both comparisons use the five GEFS leads.

| Period | Variable | Skill vs climatology | Skill vs GEFS | Spread-skill |
| --- | --- | ---: | ---: | ---: |
| 2024-01 | t2m | 63.54% | 17.06% | 0.739 |
| 2024-01 | td2m | 70.77% | 13.80% | 0.760 |
| 2024-01 | u10 | 29.89% | 8.99% | 0.776 |
| 2024-01 | v10 | 31.45% | 8.70% | 0.777 |
| 2024-01 | msl | 80.35% | 15.04% | 0.687 |
| 2024-07 | t2m | 41.97% | 39.65% | 0.940 |
| 2024-07 | td2m | 39.55% | 40.94% | 0.913 |
| 2024-07 | u10 | 11.32% | 14.84% | 0.991 |
| 2024-07 | v10 | 23.06% | 14.98% | 0.997 |
| 2024-07 | msl | 78.64% | 52.11% | 1.052 |
