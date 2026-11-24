"""Adversarial red-team tests against wrfcomp.assimilation (qc, trust, tokens).

Each test asserts that the defense HOLDS against a realistic attack (or preserves
a genuine severe-weather event). Attacks that currently defeat the defense are
marked ``@pytest.mark.xfail(strict=True, reason=...)`` without altering the
production code under audit.

All inputs are synthetic and deterministically seeded.
"""

import numpy as np
import pytest

from wrfcomp.assimilation.qc import Flag, quality_control, solar_elevation
from wrfcomp.assimilation.tokens import ObservationBatch, QCConfig

CFG = QCConfig()


def _make_batch(
    values,
    times,
    *,
    stations=None,
    var="t2m",
    lat=40.0,
    lon=-90.0,
    elevation_m=200.0,
):
    n = len(values)
    if stations is None:
        station_ids = np.repeat("SYNTH000001", n)
    else:
        station_ids = np.asarray(stations, dtype=str)
    return ObservationBatch(
        station_ids=station_ids,
        lat=np.broadcast_to(np.asarray(lat, dtype=float), (n,)),
        lon=np.broadcast_to(np.asarray(lon, dtype=float), (n,)),
        elevation_m=np.broadcast_to(np.asarray(elevation_m, dtype=float), (n,)),
        obs_time=np.asarray(times, dtype="datetime64[ns]"),
        sensor_type_id=np.zeros(n, dtype=np.int32),
        variable=np.broadcast_to(np.asarray(var, dtype=str), (n,)),
        values=np.asarray(values, dtype=np.float32),
    )


def _diurnal_series(days, *, step_minutes=60, start="2024-07-01T00", lat=40.0, lon=-90.0):
    steps = int(days * 24 * 60 // step_minutes)
    times = np.datetime64(start, "ns") + np.arange(steps) * np.timedelta64(step_minutes, "m")
    hours = np.arange(steps) * (step_minutes / 60.0)
    # Realistic July CONUS diurnal temperature wave (~288 K to ~300 K) with mild day-to-day synoptic wave.
    true_t = 294.0 + 6.0 * np.sin((hours - 15.0) * np.pi / 12.0) + 2.5 * np.sin(hours * np.pi / 84.0)
    elev = np.array([solar_elevation(t, lat, lon) for t in times])
    return times, true_t, elev


# ---------------------------------------------------------------------------
# 1. Solar-heated unshielded sensors
# ---------------------------------------------------------------------------


def test_solar_heated_sensor_penalized_when_official_bias_available():
    """Defense HOLDS: +5 K daytime radiative bias with official bias history reduces daytime weight."""
    times, true_t, elev = _diurnal_series(10)
    day = elev > CFG.day_degrees
    # Unshielded sensor runs +5 K hot whenever the sun is above 10 degrees; clean at night.
    values = true_t + np.where(day, 5.0, 0.0)
    result = quality_control(
        _make_batch(values, times),
        background=true_t,
        sigma_total=2.0,
        sigma_bg=2.0,
        reference=true_t,
        reference_time_ns=times.astype(np.int64),
        official_background_bias=0.0,
    )
    tail = slice(-48, None)
    day_weight = float(np.mean(result.weights[tail][day[tail]]))
    night_weight = float(np.mean(result.weights[tail][~day[tail]]))
    assert day_weight < 0.25 * night_weight
    assert day_weight < 0.12
    assert not result.states[("SYNTH000001", "t2m")].earned


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: Without an official station within 150 km (bias_available=False), qc.py sets "
        "learn_bias=False, disabling both bias_day/bias_night learning and mse_ema updates. "
        "An unshielded sensor with +4.5 K daytime solar heating keeps solar=1.0, reliability=1.0, "
        "and full new_prior weight (0.5) on every daytime report."
    ),
)
def test_solar_heated_sensor_without_nearby_official_history_penalized():
    """Attack: +4.5 K daytime radiative bias at a rural site with no official bias station within 150 km."""
    times, true_t, elev = _diurnal_series(10)
    day = elev > CFG.day_degrees
    values = true_t + np.where(day, 4.5, 0.0)
    # No official_background_bias supplied and no official stations in the batch -> bias_available is False.
    result = quality_control(
        _make_batch(values, times),
        background=true_t,
        sigma_total=2.0,
        sigma_bg=2.0,
    )
    tail = slice(-48, None)
    day_weight = float(np.mean(result.weights[tail][day[tail]]))
    night_weight = float(np.mean(result.weights[tail][~day[tail]]))
    assert day_weight < 0.5 * night_weight


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: update_verification only checks daily-mean reference bias <= 1*sigma_bg (2.0 K) "
        "and day-to-day bias spread, ignoring day-minus-night radiative divergence. A solar-heated "
        "sensor running +3.2 K hot every afternoon (0 K at night) has daily mean bias ~+1.7 K <= 2.0 K, "
        "so it earns the verified temperature tier (prior=1.0) and retains substantial daytime weight."
    ),
)
def test_moderate_solar_heated_sensor_denied_verified_tier():
    """Attack: Unshielded sensor with +3.2 K daytime solar error still earns the verified tier."""
    times, true_t, elev = _diurnal_series(10)
    day = elev > CFG.day_degrees
    values = true_t + np.where(day, 3.2, 0.0)
    result = quality_control(
        _make_batch(values, times),
        background=true_t,
        sigma_total=2.0,
        sigma_bg=2.0,
        reference=true_t,
        reference_time_ns=times.astype(np.int64),
        official_background_bias=0.0,
    )
    state = result.states[("SYNTH000001", "t2m")]
    assert not state.earned


# ---------------------------------------------------------------------------
# 2. Indoor sensors
# ---------------------------------------------------------------------------


def test_indoor_sensor_with_three_outdoor_buddies_decays_weight():
    """Defense HOLDS: Noisy HVAC indoor sensor with >=3 outdoor buddies triggers correlation_guard decay."""
    rng = np.random.default_rng(20240701)
    times, true_t, _ = _diurnal_series(4)
    n = len(times)
    # 1 indoor attacker + 3 honest outdoor buddies in distinct cells within 25 km.
    indoor_vals = 294.0 + 0.4 * np.sin(np.arange(n) * 1.7) + rng.normal(0.0, 0.1, n)
    b1 = true_t + rng.normal(0.0, 0.25, n)
    b2 = true_t + rng.normal(0.0, 0.25, n)
    b3 = true_t + rng.normal(0.0, 0.25, n)

    values = np.concatenate([indoor_vals, b1, b2, b3])
    all_times = np.tile(times, 4)
    stations = np.concatenate(
        [
            np.repeat("SYNTHINDR01", n),
            np.repeat("SYNTHOUTD01", n),
            np.repeat("SYNTHOUTD02", n),
            np.repeat("SYNTHOUTD03", n),
        ]
    )
    lons = np.concatenate(
        [
            np.full(n, -90.00),
            np.full(n, -89.90),
            np.full(n, -90.10),
            np.full(n, -90.00),
        ]
    )
    lats = np.concatenate(
        [
            np.full(n, 40.00),
            np.full(n, 40.00),
            np.full(n, 40.00),
            np.full(n, 40.10),
        ]
    )
    bg = np.tile(true_t, 4)
    result = quality_control(
        _make_batch(values, all_times, stations=stations, lat=lats, lon=lons),
        background=bg,
        sigma_total=2.0,
        sigma_bg=2.0,
        official_background_bias=0.0,
    )
    indoor_tail = result.weights[:n][-24:]
    assert float(np.max(indoor_tail)) < 0.05


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: trust.advance_state only triggers correlation_guard decay from buddy_r, never ref_r "
        "(trust.py:131-132). An HVAC-cycled indoor sensor without 3 buddies within 25 km (FEW_BUDDIES) "
        "keeps correlation_guard=1.0 even when paired with an official reference giving ref_r ~ 0, "
        "retaining significant weight whenever outdoor temperature passes near room temperature."
    ),
)
def test_indoor_sensor_without_three_buddies_but_with_reference_suppressed():
    """Attack: Noisy HVAC indoor sensor in sparse area (FEW_BUDDIES) with nearby official reference."""
    rng = np.random.default_rng(20240702)
    times, true_t, _ = _diurnal_series(5)
    n = len(times)
    # Thermostat-cycled indoor sensor (~294 +/- 0.4 K): never flatlines, uncorrelated with outdoor weather.
    indoor_vals = 294.0 + 0.4 * np.sin(np.arange(n) * 1.9) + rng.normal(0.0, 0.08, n)
    result = quality_control(
        _make_batch(indoor_vals, times),
        background=true_t,
        sigma_total=2.0,
        sigma_bg=2.0,
        reference=true_t,
        reference_time_ns=times.astype(np.int64),
        official_background_bias=0.0,
    )
    indoor_tail = result.weights[-24:]
    assert float(np.max(indoor_tail)) < 0.05


# ---------------------------------------------------------------------------
# 3. Frozen anemometer
# ---------------------------------------------------------------------------


def test_frozen_anemometer_nonzero_at_15min_cadence_rejected():
    """Defense HOLDS: Non-zero anemometer stuck at 4.0 m/s (15-min cadence) vs 9.0 m/s ref is flatlined."""
    times = np.datetime64("2024-07-12T06", "ns") + np.arange(8) * np.timedelta64(15, "m")
    result = quality_control(
        _make_batch(np.full(8, 4.0), times, var="wind_speed"),
        background=9.0,
        sigma_total=2.0,
        sigma_bg=2.0,
        reference=9.0,
    )
    assert bool(result.qc_flags[5] & Flag.FLATLINE)
    assert result.weights[5] == 0.0
    assert result.weights[-1] == 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: temporal_gate exempts abs(value) <= calm_wind_ms (0.5 m/s) unconditionally "
        "(qc.py:52) without checking reference, buddies, or background; Flag.SPATIAL is also "
        "excluded from hard_fail (qc.py:298), and wind_speed retains learn_bias=True so bias_day "
        "absorbs the -10 m/s deficit. A seized 0.0 m/s anemometer in 10 m/s wind never flatlines "
        "and retains positive weight."
    ),
)
def test_frozen_anemometer_seized_at_zero_in_high_wind_rejected():
    """Attack: Anemometer frozen/seized at 0.0 m/s for 6 hours while ambient wind blows at 10 m/s."""
    times = np.datetime64("2024-07-12T06", "ns") + np.arange(24) * np.timedelta64(15, "m")
    wind_true = 10.0 + 1.5 * np.sin(np.arange(24) * 0.4)
    result = quality_control(
        _make_batch(np.zeros(24), times, var="wind_speed"),
        background=wind_true,
        sigma_total=2.0,
        sigma_bg=2.0,
        reference=wind_true,
    )
    assert bool(result.qc_flags[-1] & Flag.FLATLINE)
    assert result.weights[-1] == 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: Both temporal_gate (qc.py:53) and advance_state (trust.py:137) require consecutive "
        "reports to be <= step_minutes (15 min) apart to advance constant_since_ns and trigger "
        "wind FLATLINE. A frozen anemometer stuck at 5.0 m/s reporting every 20 minutes (or hourly) "
        "never triggers Flag.FLATLINE and keeps positive weight indefinitely."
    ),
)
def test_frozen_anemometer_reporting_slower_than_15min_rejected():
    """Attack: Anemometer stuck at 5.0 m/s reporting every 20 minutes for 8 hours vs 10 m/s ref."""
    times = np.datetime64("2024-07-12T06", "ns") + np.arange(24) * np.timedelta64(20, "m")
    wind_true = 10.0 + 1.0 * np.sin(np.arange(24) * 0.3)
    result = quality_control(
        _make_batch(np.full(24, 5.0), times, var="wind_speed"),
        background=wind_true,
        sigma_total=2.0,
        sigma_bg=2.0,
        reference=wind_true,
    )
    assert bool(result.qc_flags[-1] & Flag.FLATLINE)
    assert result.weights[-1] == 0.0


# ---------------------------------------------------------------------------
# 4. Slow drift
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: bias_day and bias_night EMA (alpha=0.04) have no magnitude cap and smoothly absorb "
        "a uniform slow drift (+0.5 K/day -> +8 K error), keeping residual~0, huber=1, reliability~1, "
        "and solar=1 (since |bias_day - bias_night|~0). Even when Flag.SPATIAL fires against 3 honest "
        "buddies, Flag.SPATIAL is excluded from hard_fail (qc.py:298) and does not reduce weight."
    ),
)
def test_slow_uniform_drift_to_large_error_downweighted():
    """Attack: Uniform +0.5 K/day drift for 16 days (+8 K error) with 3 honest buddies."""
    rng = np.random.default_rng(20240704)
    times, true_t, _ = _diurnal_series(16)
    n = len(times)
    drift = np.linspace(0.0, 8.0, n)
    drifter = true_t + drift + rng.normal(0.0, 0.15, n)
    b1 = true_t + rng.normal(0.0, 0.2, n)
    b2 = true_t + rng.normal(0.0, 0.2, n)
    b3 = true_t + rng.normal(0.0, 0.2, n)

    values = np.concatenate([drifter, b1, b2, b3])
    all_times = np.tile(times, 4)
    stations = np.concatenate(
        [
            np.repeat("SYNTHDRFT01", n),
            np.repeat("SYNTHGOOD01", n),
            np.repeat("SYNTHGOOD02", n),
            np.repeat("SYNTHGOOD03", n),
        ]
    )
    lons = np.concatenate([np.full(n, -90.0), np.full(n, -89.9), np.full(n, -90.1), np.full(n, -90.0)])
    lats = np.concatenate([np.full(n, 40.0), np.full(n, 40.0), np.full(n, 40.0), np.full(n, 40.1)])
    bg = np.tile(true_t, 4)

    result = quality_control(
        _make_batch(values, all_times, stations=stations, lat=lats, lon=lons),
        background=bg,
        sigma_total=2.0,
        sigma_bg=2.0,
        reference=bg,
        reference_time_ns=all_times.astype(np.int64),
        official_background_bias=0.0,
    )
    drifter_tail_weights = result.weights[:n][-24:]
    drifter_tail_flags = result.qc_flags[:n][-24:]
    # Spatial gate notices the +8 K departure, yet weight must also collapse once drifted by +8 K.
    assert np.all(drifter_tail_flags & Flag.SPATIAL)
    assert float(np.mean(drifter_tail_weights)) < 0.10


# ---------------------------------------------------------------------------
# 5. Airport-scraping copycat (copied nearby official obs plus small noise)
# ---------------------------------------------------------------------------


def test_exact_airport_copycat_caught_and_zeroed():
    """Defense HOLDS: Exact copy of official reference triggers Flag.COPYCAT and zero weight."""
    times, ref, _ = _diurnal_series(3)
    result = quality_control(
        _make_batch(ref, times),
        background=ref,
        sigma_total=2.0,
        sigma_bg=2.0,
        reference=ref,
        reference_time_ns=times.astype(np.int64),
        official_background_bias=0.0,
    )
    assert bool(result.qc_flags[-1] & Flag.COPYCAT)
    assert result.weights[-1] == 0.0
    assert not result.states[("SYNTH000001", "t2m")].earned


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: Copycat detection requires ref_rmse <= copy_residual_sigma * sigma_bg "
        "(0.01 * 2.0 = 0.02 K, trust.py:134). Adding tiny Gaussian noise (sigma=0.05 K) to scraped "
        "airport observations evades Flag.COPYCAT completely AND allows the scraper to earn the "
        "verified temperature tier (prior=1.0, weight > 0.9)."
    ),
)
def test_airport_copycat_with_small_noise_caught_and_denied_verified():
    """Attack: Scraper copies nearby official airport temperature plus tiny 0.05 K Gaussian noise."""
    rng = np.random.default_rng(20240705)
    times, ref, _ = _diurnal_series(9)
    scraped = ref + rng.normal(0.0, 0.05, len(times))
    result = quality_control(
        _make_batch(scraped, times),
        background=ref,
        sigma_total=2.0,
        sigma_bg=2.0,
        reference=ref,
        reference_time_ns=times.astype(np.int64),
        official_background_bias=0.0,
    )
    state = result.states[("SYNTH000001", "t2m")]
    assert bool(result.qc_flags[-1] & Flag.COPYCAT)
    assert not state.earned
    assert result.weights[-1] == 0.0


# ---------------------------------------------------------------------------
# 6. Colluding cluster of fake neighbours
# ---------------------------------------------------------------------------


def test_same_cell_colluding_cluster_cannot_corroborate_fake_spike():
    """Defense HOLDS: 4 fake citizens in the same tenth-degree cell fail independence checks."""
    t = np.datetime64("2024-07-12T18", "ns")
    batch = _make_batch(
        values=[280.0, 280.0, 280.0, 280.0],
        times=[t] * 4,
        stations=["SYNTHFAKE01", "SYNTHFAKE02", "SYNTHFAKE03", "SYNTHFAKE04"],
        lat=[40.00, 40.01, 40.02, 40.03],
        lon=[-90.00, -90.01, -90.02, -90.03],
    )
    result = quality_control(
        batch,
        background=290.0,
        sigma_total=1.0,
        sigma_bg=2.0,
        official_background_bias=0.0,
    )
    assert np.all(result.qc_flags & Flag.FEW_BUDDIES)
    assert np.all(result.huber == 0.0)
    assert np.all(result.weights == 0.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: Cell independence uses np.rint(lon * 10) (qc.py:110). Four Sybil citizen stations "
        "placed only ~1.7 km apart straddling a 0.1-degree boundary (lon -90.04 and -90.06) are "
        "treated as independent, letting them corroborate each other's fabricated >8-sigma spike "
        "in background_gate with positive weight despite an honest nearby official station."
    ),
)
def test_cross_cell_boundary_sybil_cluster_cannot_corroborate_fake_spike():
    """Attack: 4 Sybil stations ~1.7 km apart across a 0.1-deg grid line fake a -10 sigma storm."""
    t = np.datetime64("2024-07-12T18", "ns")
    # 4 fake citizens straddling lon=-90.05 (rounds to -900 vs -901) + 1 honest official station nearby.
    batch = _make_batch(
        values=[280.0, 280.0, 280.0, 280.0, 290.0],
        times=[t] * 5,
        stations=["SYNTHSYBL01", "SYNTHSYBL02", "SYNTHSYBL03", "SYNTHSYBL04", "SYNTHOFFC01"],
        lat=[40.00, 40.005, 40.00, 40.005, 40.02],
        lon=[-90.04, -90.04, -90.06, -90.06, -90.05],
    )
    result = quality_control(
        batch,
        background=290.0,
        sigma_total=1.0,
        sigma_bg=2.0,
        official=[False, False, False, False, True],
        official_background_bias=0.0,
    )
    assert np.all(result.weights[:4] == 0.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: spatial_series weights each official station at 2x and each citizen at 1x (qc.py:84). "
        "Five colluding fake citizens placed ~1.7 km apart across a 0.1-deg boundary outvote two "
        "genuine official stations (5 > 2*2), causing both official stations to be flagged "
        "Flag.SPATIAL while all five fake citizens pass spatial QC unflagged."
    ),
)
def test_sybil_cluster_cannot_outvote_official_stations_in_spatial_gate():
    """Attack: 5 Sybil citizens across a 0.1-deg boundary outvote 2 official stations in spatial QC."""
    t = np.datetime64("2024-07-12T18", "ns")
    # 5 fakes at 280 K straddling lon=-90.05, vs 2 honest official stations at true 290 K.
    batch = _make_batch(
        values=[280.0, 280.0, 280.0, 280.0, 280.0, 290.0, 290.0],
        times=[t] * 7,
        stations=[
            "SYNTHSYBL01",
            "SYNTHSYBL02",
            "SYNTHSYBL03",
            "SYNTHSYBL04",
            "SYNTHSYBL05",
            "SYNTHOFFC01",
            "SYNTHOFFC02",
        ],
        lat=[40.00, 40.005, 40.01, 40.00, 40.005, 40.02, 39.98],
        lon=[-90.04, -90.04, -90.04, -90.06, -90.06, -90.05, -90.05],
    )
    result = quality_control(
        batch,
        background=290.0,
        sigma_total=1.0,
        sigma_bg=2.0,
        official=[False, False, False, False, False, True, True],
        official_background_bias=0.0,
    )
    # Honest official stations must NOT be spatially flagged, and fake cluster must be flagged.
    assert not bool(result.qc_flags[5] & Flag.SPATIAL)
    assert not bool(result.qc_flags[6] & Flag.SPATIAL)
    assert np.all(result.qc_flags[:5] & Flag.SPATIAL)


# ---------------------------------------------------------------------------
# 7. Real severe-storm spike seen by two good neighbours (MUST BE KEPT)
# ---------------------------------------------------------------------------


def test_real_severe_storm_spike_seen_by_two_good_neighbours_is_kept():
    """Defense HOLDS: Genuine >8-sigma severe-storm cold pool & squall seen by 2 good neighbours is kept."""
    t0 = np.datetime64("2024-07-12T17", "ns")
    t1 = np.datetime64("2024-07-12T18", "ns")
    # Hourly cadence: pre-storm normal (296 K) followed by intense -12 K cold-pool outflow (284 K),
    # seen by target citizen and two independent good neighbours across distinct cells within 25 km.
    stations = ["SYNTHGOOD01", "SYNTHGOOD02", "SYNTHGOOD03"]
    lats = [40.0, 40.1, 39.9]
    lons = [-90.0, -89.9, -90.1]
    batch = _make_batch(
        values=[296.0, 296.2, 295.8, 284.0, 284.5, 284.2],
        times=[t0, t0, t0, t1, t1, t1],
        stations=stations * 2,
        lat=lats * 2,
        lon=lons * 2,
    )
    result = quality_control(
        batch,
        background=296.0,
        sigma_total=1.0,
        sigma_bg=2.0,
        official=[False, False, True, False, False, True],
        official_background_bias=0.0,
    )
    # Target citizen at storm hour (row 3): z = -12 (beyond gross_cut=8), corroborated by 2 neighbours.
    assert not bool(result.qc_flags[3] & (Flag.STATIC | Flag.TEMPORAL | Flag.COPYCAT))
    assert result.huber[3] > 0.0
    assert result.weights[3] > 0.0

    # Also verify a genuine severe convective wind gust spike (+18 m/s above background, z=+9).
    wind_batch = _make_batch(
        values=[23.0, 22.0, 24.0],
        times=[t1, t1, t1],
        stations=stations,
        var="wind_speed",
        lat=lats,
        lon=lons,
    )
    wind_res = quality_control(
        wind_batch,
        background=5.0,
        sigma_total=2.0,
        sigma_bg=2.0,
        official=[False, False, True],
    )
    assert not bool(wind_res.qc_flags[0] & (Flag.STATIC | Flag.TEMPORAL | Flag.COPYCAT))
    assert wind_res.huber[0] > 0.0
    assert wind_res.weights[0] > 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BROKEN: temporal_gate hard-rejects any t2m step > 10 K within <= 15 minutes (qc.py:45-46, "
        "263, 298) before spatial/buddy corroboration, excluding all corroborated stations from "
        "eligible and zeroing their weights. A genuine supercell outflow boundary dropping -12 K "
        "in 15 minutes across 3 independent stations (including an official site) is completely lost."
    ),
)
def test_real_rapid_15min_severe_storm_drop_seen_by_two_good_neighbours_is_kept():
    """False-positive attack: Real -12 K outflow drop in 15 min corroborated by 2 good neighbours."""
    t0 = np.datetime64("2024-07-12T18:00", "ns")
    t1 = np.datetime64("2024-07-12T18:15", "ns")
    stations = ["SYNTHGOOD01", "SYNTHGOOD02", "SYNTHGOOD03"]
    lats = [40.0, 40.1, 39.9]
    lons = [-90.0, -89.9, -90.1]
    batch = _make_batch(
        values=[298.0, 298.1, 297.9, 286.0, 286.3, 285.8],
        times=[t0, t0, t0, t1, t1, t1],
        stations=stations * 2,
        lat=lats * 2,
        lon=lons * 2,
    )
    result = quality_control(
        batch,
        background=298.0,
        sigma_total=1.0,
        sigma_bg=2.0,
        official=[False, False, True, False, False, True],
        official_background_bias=0.0,
    )
    assert result.weights[3] > 0.0
    assert result.huber[3] > 0.0
