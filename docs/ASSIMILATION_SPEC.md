# Station quality control

Call `quality_control` with an observation batch, the previous forecast's background at each report, a total innovation uncertainty and a background uncertainty, all in the variable's native units. Supply actual measurement times and only reports available by the assimilation cutoff. For hourly truth input, reconstruct each measurement time as `valid_time + X_offset_min` before applying that cutoff. The driver processes UTC timestamps in order and returns flags, Huber and final weights, buddy medians and updated state in the original row order. Raw columns are copied at ingestion and read-only. No coordinates are logged.

| Gate | Flag bits | Rule |
| --- | --- | --- |
| Static | 1 | Check physical ranges and simultaneous station reports for dewpoint above temperature or gust below wind speed. Compare station pressure to elevation-adjusted standard pressure. Mean sea-level pressure has its own range. |
| Temporal | 2, plus 64 for flatline | Reject rapid changes inside the configured interval (temperature defaults to 10 K per 15 minutes) and constant values over six consecutive reports. Zero precipitation is exempt from flatline rejection. |
| Spatial | 4 for an outlier, 16 for insufficient coverage | Compare against simultaneous distinct stations within 25 km that passed static and temporal QC. Adjust temperature to target elevation at 6.5 K/km. Flag deviations above four robust standard deviations from the median, using scaled MAD with a floor. Fewer than three buddies leaves the spatial test unapplied. Both flags are diagnostic. |
| Background | 8 | Compute innovation minus the prior daytime or nighttime bias, divided by total uncertainty. UTC time and position give approximate solar elevation. Above 10 degrees counts as day. Huber weight is 1 through 2.5 sigma, then 2.5 divided by absolute z. Beyond 8 sigma, weight is zero unless two distinct simultaneous buddies exceed 3 sigma with matching sign. |
| Trust | 32 | Flag a suspected copy when the seven-day reference correlation exceeds 0.998 with near-zero RMS difference. Reference values must come from an independently identified feed. Copycats have zero weight. |

For each station and variable, the driver retains separate day/night innovation EMAs, an EMA of squared bias-corrected background residuals, the recent temporal window and timestamped pairs for correlation. Persist each `StationState` via `dataclasses.asdict` and JSON, then restore with `StationState(**record)`. Supply the returned states on the next call. Submit each station-variable timestamp once. Supply the official tier explicitly.

Final weight is `clip(tier_prior * w_huber * R * S * C, 0, 1)`, with priors 1 for official and 0.5 for new stations. Static or temporal failures receive zero weight. Reliability is `exp(-sqrt(mse_ema) / sigma_bg)`. Daytime temperature uses `S = exp(-max(0, abs(bias_day - bias_night) - 1) / 2)` once both biases exist. Other reports use S = 1. The seven-day correlation guard starts at 1, decays by 0.9 per update below 0.5 correlation with the buddy median, and reaches zero for a copycat. A constant station against varying buddies counts as zero correlation. The latest final weight is the QoD score. Thresholds and native-unit bounds are in frozen `QCConfig`.

HealDA integration and the SDA refiner are deferred. The QC path uses CPU NumPy and synthetic tests.
