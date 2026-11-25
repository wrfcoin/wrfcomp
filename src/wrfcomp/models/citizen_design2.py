"""Phase 3 Design 2 (M = 2): Track A upwind+tempered C2, Track B 5-fold LOAO, and relay audit.

Implements the frozen Phase 3 Design 2 specification without modifying Design 1's frozen
``wrfcomp.models.citizen_arms`` or ``wrfcomp.assimilation.phase3_qc`` modules:

- **Change 1 (upwind neighbourhood):** haversine-scaled displacement vectors and a lead-specific
  upwind Gaussian shifted along prior-cycle raw GEFS 10 m wind (issued at ``T - 12 h``, lead
  ``12 h``, valid at ``T``; calm fallback ``(0, 0)`` on January 1 00Z) with transport multiplier
  ``alpha_adv = 1.5``, displacement capped at radius ``R``.
- **Change 2 (tempered QC trust):** ``w_eff = sqrt(w_c)`` for ``w_c > 0`` (zero otherwise),
  preserving Design 1's exact report-selection order and ``COVERED`` station set.
- **Track B (simulated multi-airport outage):** deterministic maximally-separated 5-fold outer
  exclusion (all held-out stations in fold ``f`` deleted from ``truth_in`` before computing any
  official predictor, historical training row, or pooled gain) paired with the official-free
  ``phase3_qc_loao`` extract and target-independent ``cell(c) != cell(s)`` exclusion.
- **Post-freeze relay diagnostics:** aggregate-only nearest-reference flag share and target-paired
  distinct-official-measurement audit via frozen ``trial.official_evidence`` + ``suspected_copy``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from wrfcomp.assimilation.qc import cells, distances_km
from wrfcomp.assimilation.trust import suspected_copy
from wrfcomp.models import citizen_arms as arms
from wrfcomp.models import gefs_station_v2 as v2

MODEL_CITIZEN_D2 = "gefs-station-corrected-citizen-d2"
MODEL_LOAO_BASE = "gefs-loao-official"
MODEL_LOAO_CITIZEN = "gefs-loao-citizen"
VERSION = "1"

VARIABLES = arms.VARIABLES  # ("t2m", "td2m")
LEAD_HOURS = arms.LEAD_HOURS  # (1, 3, 6)
ALL_LEAD_HOURS = (1, 3, 6, 12, 24, 48)
PRIOR_LEAD_HOURS = arms.OWN_LEAD_HOURS  # 12
ALPHA_ADV = 1.5
EARTH_RADIUS_KM = 6371.0088

RADIUS_KM = arms.RADIUS_KM  # 50.0
DISTANCE_SCALE_KM = arms.DISTANCE_SCALE_KM  # 25.0
MAX_HEIGHT_DIFFERENCE_M = arms.MAX_HEIGHT_DIFFERENCE_M  # 200.0
OFFICIAL_RADIUS_KM = 150.0
OFFICIAL_DISTANCE_SCALE_KM = 75.0
MIN_OFFICIAL_NEIGHBOURS = 1

USUAL_DAYS = arms.USUAL_DAYS  # 30
USUAL_MIN = arms.USUAL_MIN  # 10
CLIP_K = arms.CLIP_K  # 5.0
MIN_NEIGHBOURS = arms.MIN_NEIGHBOURS  # 3
MIN_CELLS = arms.MIN_CELLS  # 2
GAIN_WINDOW = arms.GAIN_WINDOW  # 30 D
MAX_OFFSET_MIN = arms.MAX_OFFSET_MIN  # 15
N_FOLDS = 5
LAPSE_K_PER_KM = 6.5

DAY = arms.DAY
HOUR = arms.HOUR
MINUTE = arms.MINUTE


# ----------------------------------------------------------------------------- geometry & weights


def haversine_displacement_km(
    s_lat: float, s_lon: float, c_lat: np.ndarray, c_lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Haversine-scaled local displacement ``(rx, ry, d_gc)`` in km from station ``s`` to ``c``.

    Scaling the tangent-plane vector to exact great-circle distance ``d_gc`` guarantees that at
    calm wind ``(px, py) = (0, 0)``, ``rx^2 + ry^2 == d_gc^2`` to machine precision.
    """
    c_lat = np.asarray(c_lat, dtype=np.float64)
    c_lon = np.asarray(c_lon, dtype=np.float64)
    d_gc = distances_km(float(s_lat), float(s_lon), c_lat, c_lon).astype(np.float64)
    lat_r = np.radians(float(s_lat))
    dx_tp = EARTH_RADIUS_KM * np.cos(lat_r) * np.radians(c_lon - float(s_lon))
    dy_tp = EARTH_RADIUS_KM * np.radians(c_lat - float(s_lat))
    norm_tp = np.hypot(dx_tp, dy_tp)
    valid = (d_gc > 0.0) & (norm_tp > 0.0)
    scale = np.where(valid, d_gc / np.where(valid, norm_tp, 1.0), 0.0)
    return dx_tp * scale, dy_tp * scale, d_gc


def upwind_target_km(
    u10: float,
    v10: float,
    lead_hours: float,
    radius_km: float,
    *,
    alpha_adv: float = ALPHA_ADV,
) -> tuple[float, float]:
    """Upwind back-trajectory target ``(px, py)`` in km, capped at ``radius_km``."""
    if not (np.isfinite(u10) and np.isfinite(v10)):
        return 0.0, 0.0
    speed = float(np.hypot(u10, v10))
    if speed == 0.0:
        return 0.0, 0.0
    d_up = min(3.6 * float(lead_hours) * float(alpha_adv) * speed, float(radius_km))
    return -d_up * (float(u10) / speed), -d_up * (float(v10) / speed)


def upwind_gaussian_weight(
    rx: np.ndarray,
    ry: np.ndarray,
    d_gc: np.ndarray,
    px: float,
    py: float,
    scale_km: float,
) -> np.ndarray:
    """Upwind Gaussian weight; reduces identically to Design 1's haversine weight at calm wind."""
    if px == 0.0 and py == 0.0:
        return np.exp(-np.square(d_gc / scale_km) / 2.0)
    sq = np.square(rx - px) + np.square(ry - py)
    return np.exp(-sq / (2.0 * scale_km * scale_km))


def temper_trust_weight(weight: np.ndarray) -> np.ndarray:
    """Change 2: parameter-free square-root tempering ``sqrt(w)`` for ``w > 0``, else ``0``."""
    w = np.asarray(weight, dtype=np.float64)
    return np.where(w > 0.0, np.sqrt(np.maximum(w, 0.0)), 0.0)


@dataclass(frozen=True)
class OrientedPairs:
    """Eligible ``(citizen, official)`` pairs with exact haversine-scaled displacements."""

    citizen: np.ndarray
    official: np.ndarray
    distance_km: np.ndarray
    rx_km: np.ndarray
    ry_km: np.ndarray
    cell: np.ndarray  # tenth-degree cell of the citizen station


def oriented_citizen_pairs(
    c_lat,
    c_lon,
    c_height,
    o_lat,
    o_lon,
    o_height,
    *,
    radius_km: float = RADIUS_KM,
    max_height_m: float = MAX_HEIGHT_DIFFERENCE_M,
    exclude_same_cell: bool = False,
) -> OrientedPairs:
    """Build eligible ``(citizen, official)`` pairs with haversine-scaled displacement vectors."""
    c_lat, c_lon, c_height = (np.asarray(a, dtype=np.float64) for a in (c_lat, c_lon, c_height))
    o_lat, o_lon, o_height = (np.asarray(a, dtype=np.float64) for a in (o_lat, o_lon, o_height))
    c_cells = cells(c_lat, c_lon) if c_lat.size else np.empty(0, dtype=int)
    o_cells = cells(o_lat, o_lon) if o_lat.size else np.empty(0, dtype=int)
    cit, off, dist, rxs, rys = [], [], [], [], []
    for s in range(len(o_lat)):
        if not c_lat.size:
            continue
        rx, ry, km = haversine_displacement_km(float(o_lat[s]), float(o_lon[s]), c_lat, c_lon)
        mask = (km <= radius_km) & (np.abs(c_height - float(o_height[s])) <= max_height_m)
        if exclude_same_cell:
            mask &= c_cells != o_cells[s]
        ok = np.flatnonzero(mask)
        cit.append(ok)
        off.append(np.full(ok.size, s, dtype=int))
        dist.append(km[ok])
        rxs.append(rx[ok])
        rys.append(ry[ok])
    citizen = np.concatenate(cit).astype(int) if cit else np.empty(0, int)
    return OrientedPairs(
        citizen=citizen,
        official=np.concatenate(off).astype(int) if off else np.empty(0, int),
        distance_km=np.concatenate(dist) if dist else np.empty(0, float),
        rx_km=np.concatenate(rxs) if rxs else np.empty(0, float),
        ry_km=np.concatenate(rys) if rys else np.empty(0, float),
        cell=c_cells[citizen] if citizen.size else np.empty(0, int),
    )


def _one_per_token(token_ids: np.ndarray | None, pairs_cit: np.ndarray, use: np.ndarray, age, weights):
    """Design 1 tie-break: newest exact age_ns, then larger weight, then lower row."""
    if token_ids is None or use.size < 2:
        return np.arange(use.size)
    tok = token_ids[pairs_cit[use]]
    order = np.lexsort((np.arange(use.size), -weights, age[pairs_cit[use]], tok))
    first = np.r_[True, tok[order][1:] != tok[order][:-1]]
    return np.sort(order[first])


# ----------------------------------------------------------------------------- Track A (C2)


@dataclass
class _TrackAIssue:
    init: np.datetime64
    own: dict  # var -> (station,)
    citizen: dict  # var -> (n_leads, station)
    mean_a: dict  # var -> (n_leads, station)
    mean_b: dict  # var -> (n_leads, station)
    mean_own_lead: dict  # var -> (station,)
    difference: dict  # var -> (pair,)


@dataclass
class TrackADiagnostics:
    init: str
    phi: dict = field(default_factory=dict)
    gamma: dict = field(default_factory=dict)
    own_defined: dict = field(default_factory=dict)
    citizen_defined: dict = field(default_factory=dict)
    weight_total: dict = field(default_factory=dict)
    weight_flagged: dict = field(default_factory=dict)


class TrackAReplay:
    """Design 2 Track A same-year replay producing B (identical to v4) and C2 (upwind + sqrt(w))."""

    def __init__(self, year: int, truth: xr.Dataset, pairs: OrientedPairs, flagged=None, token=None):
        self.year = year
        self.start, self.end = v2.year_bounds(year)
        if truth.valid_time.values.min() < self.start:
            raise ValueError("truth before the start of the year must not be passed in")
        self.truth = truth
        self.pairs = pairs
        self.flagged = np.zeros(0, bool) if flagged is None else np.asarray(flagged, dtype=bool)
        self.token = None if token is None else np.unique(np.asarray(token), return_inverse=True)[1]
        order = np.argsort(pairs.official, kind="stable")
        self.by_official = (
            np.split(order, np.flatnonzero(np.diff(pairs.official[order])) + 1) if order.size else []
        )
        self.history: deque[_TrackAIssue] = deque()
        self.citizen_defined_count: np.ndarray | None = None

    def _reading(self, var: str, t) -> np.ndarray:
        size = self.truth.sizes["station_id"]
        if t not in self.truth.indexes["valid_time"]:
            return np.full(size, np.nan)
        slot = self.truth.sel(valid_time=t)
        value = slot[var].values.astype(np.float64)
        offset = slot[var + "_offset_min"].values.astype(np.float64)
        ok = np.isfinite(value) & (offset >= -MAX_OFFSET_MIN) & (offset <= 0)
        ok &= offset >= (self.start - t) / MINUTE
        return np.where(ok, value, np.nan)

    def _neighbourhood(self, var, t, reading, value, weight, age, u10, v10, diag):
        p = self.pairs
        size = self.truth.sizes["station_id"]
        n_leads = len(LEAD_HOURS)
        result = np.full((n_leads, size), np.nan)
        if not p.citizen.size:
            diag.weight_total[var], diag.weight_flagged[var] = 0.0, 0.0
            return result, np.empty(0)
        y, w = value[p.citizen], weight[p.citizen]
        with np.errstate(invalid="ignore"):
            eligible = np.isfinite(y) & (w > 0.0) & np.isfinite(reading[p.official])
        difference = np.where(eligible, y - reading[p.official], np.nan)
        earlier = [
            item.difference[var]
            for item in self.history
            if t - USUAL_DAYS * DAY <= item.init <= t - DAY
            and (t - item.init) % DAY == np.timedelta64(0, "ns")
        ]
        if not earlier:
            diag.weight_total[var], diag.weight_flagged[var] = 0.0, 0.0
            return result, difference
        stack = np.stack(earlier)
        count = np.isfinite(stack).sum(axis=0)
        usual = np.full(p.citizen.size, np.nan)
        enough = count >= USUAL_MIN
        if enough.any():
            usual[enough] = np.nanmedian(stack[:, enough], axis=0)
        departure = np.clip(difference - usual, -CLIP_K, CLIP_K)
        w_eff = temper_trust_weight(w)
        total = flagged = 0.0
        for rows in self.by_official:
            use_base = rows[np.isfinite(departure[rows])]
            if use_base.size < MIN_NEIGHBOURS:
                continue
            s = int(p.official[use_base[0]])
            for li, lead_h in enumerate(LEAD_HOURS):
                px, py = upwind_target_km(float(u10[s]), float(v10[s]), lead_h, RADIUS_KM)
                geo_w = upwind_gaussian_weight(
                    p.rx_km[use_base], p.ry_km[use_base], p.distance_km[use_base], px, py, DISTANCE_SCALE_KM
                )
                weights = w_eff[use_base] * geo_w
                keep = _one_per_token(self.token, p.citizen, use_base, age, weights)
                use, weights = use_base[keep], weights[keep]
                if use.size < MIN_NEIGHBOURS or np.unique(p.cell[use]).size < MIN_CELLS:
                    continue
                result[li, s] = arms.weighted_lower_median(departure[use], weights)
                if li == 1:  # record aggregate diagnostic on 3 h lead
                    total += float(weights.sum())
                    if self.flagged.size:
                        flagged += float(weights[self.flagged[p.citizen[use]]].sum())
        diag.weight_total[var], diag.weight_flagged[var] = total, flagged
        return result, difference

    def _records(self, var, t, lead_index, lead):
        past = [
            item
            for item in self.history
            if item.init < t and (t - item.init) % DAY == np.timedelta64(0, "ns")
        ]
        past = [item for item in past if item.init + lead + MAX_OFFSET_MIN * MINUTE > t - GAIN_WINDOW]
        if not past:
            return None
        labels = np.array([item.init + lead for item in past], dtype="datetime64[ns]")
        slots = self.truth.sel(valid_time=labels)
        y = slots[var].values.astype(np.float64)
        offsets = slots[var + "_offset_min"].values.astype(np.float64)
        age_min = ((t - labels) / MINUTE)[:, None] - offsets
        ok = (
            np.isfinite(y)
            & (np.abs(offsets) <= MAX_OFFSET_MIN)
            & (age_min >= 0)
            & (age_min < GAIN_WINDOW / MINUTE)
            & (age_min <= (t - self.start) / MINUTE)
        )
        error_a = y - np.stack([item.mean_a[var][lead_index] for item in past])
        error_b = y - np.stack([item.mean_b[var][lead_index] for item in past])
        own = np.stack([item.own[var] for item in past])
        citizen = np.stack([item.citizen[var][lead_index] for item in past])
        return ok, error_a, error_b, own, citizen

    def issue(
        self,
        a: xr.Dataset,
        snapshot: dict,
        wind_prev: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[xr.Dataset, xr.Dataset, TrackADiagnostics]:
        """Step one v3 issue. ``wind_prev`` is ``(u10_mean, v10_mean)`` from raw GEFS ``T-12h@12h``
        or ``None`` on January 1 00Z (calm fallback ``(0, 0)``)."""
        if (
            a.sizes["init_time"] != 1
            or a.attrs.get("model_id") != "gefs-station-corrected"
            or a.attrs.get("model_version") != "3"
        ):
            raise ValueError("expected one issue of gefs-station-corrected version 3")
        t = a.init_time.values[0]
        if not self.start <= t or t + v2.LEADS[-1] >= self.end:
            raise ValueError("issue outside the year")
        if self.history and t <= self.history[-1].init:
            raise ValueError("initializations must increase")
        if not a.indexes["station_id"].equals(self.truth.indexes["station_id"]):
            raise ValueError("truth and forecast station grids differ")
        stations = a.station_id.values
        if not np.array_equal(np.sort(stations), stations):
            raise ValueError("stations must be in ascending id order")
        size = len(stations)
        if wind_prev is None:
            u10 = np.zeros(size, dtype=np.float64)
            v10 = np.zeros(size, dtype=np.float64)
        else:
            u10 = np.asarray(wind_prev[0], dtype=np.float64)
            v10 = np.asarray(wind_prev[1], dtype=np.float64)
        lead_values = a.lead_time.values
        lead_index = [int(np.flatnonzero(lead_values == h * HOUR)[0]) for h in LEAD_HOURS]
        own_index = int(np.flatnonzero(lead_values == PRIOR_LEAD_HOURS * HOUR)[0])
        while self.history and self.history[0].init < t - (USUAL_DAYS + 2) * DAY:
            self.history.popleft()
        previous = next((item for item in self.history if item.init == t - PRIOR_LEAD_HOURS * HOUR), None)
        diag = TrackADiagnostics(init=str(np.datetime_as_string(t, unit="h")))
        record = _TrackAIssue(t, {}, {}, {}, {}, {}, {})
        b, c2 = a.copy(deep=False), a.copy(deep=False)
        b.attrs = {**a.attrs, "model_id": arms.MODEL_OWN, "model_version": VERSION}
        c2.attrs = {**a.attrs, "model_id": MODEL_CITIZEN_D2, "model_version": VERSION}
        new_b: dict[str, np.ndarray] = {}
        shifts_c: dict[str, list[np.ndarray]] = {}
        for var in VARIABLES:
            members = a[var].values[0].astype(np.float64)
            mean = members.mean(axis=-1)
            reading = self._reading(var, t)
            own = reading - previous.mean_own_lead[var] if previous is not None else np.full(size, np.nan)
            value, weight, *rest = snapshot[var]
            value, weight = np.asarray(value, float), np.asarray(weight, float)
            age = np.asarray(rest[0]).astype(np.int64) if rest else np.zeros(len(value), dtype=np.int64)
            citizen, difference = self._neighbourhood(var, t, reading, value, weight, age, u10, v10, diag)
            out_b = members.copy()
            diag.phi[var], diag.gamma[var] = [], []
            shifts_c[var] = []
            for k, (j, hours) in enumerate(zip(lead_index, LEAD_HOURS, strict=True)):
                phi = gamma = 0.0
                records = self._records(var, t, k, hours * HOUR)
                if records is not None:
                    ok, error_a, error_b, past_own, past_citizen = records
                    use = ok & np.isfinite(past_own)
                    phi = arms.gain(past_own[use], error_a[use], size)
                    use = ok & np.isfinite(past_citizen)
                    gamma = arms.gain(past_citizen[use], error_b[use], size)
                shift_b = np.where(np.isfinite(own), phi * own, 0.0)
                shift_cit = np.where(np.isfinite(citizen[k]), gamma * citizen[k], 0.0)
                out_b[j] = members[j] + shift_b[:, None]
                shifts_c[var].append(shift_cit)
                diag.phi[var].append(phi)
                diag.gamma[var].append(gamma)
            new_b[var] = out_b.astype(np.float32)
            record.own[var], record.citizen[var], record.difference[var] = own, citizen, difference
            record.mean_a[var] = mean[lead_index]
            record.mean_own_lead[var] = mean[own_index]
            diag.own_defined[var] = int(np.isfinite(own).sum())
            diag.citizen_defined[var] = int(np.isfinite(citizen[1]).sum())  # 3 h lead
        # Finish control B first (including td2m <= t2m cap), then add citizen shifts to finished B.
        for j in lead_index:
            new_b["td2m"][j] = np.minimum(new_b["td2m"][j], new_b["t2m"][j])
        for var in VARIABLES:
            mean_b = np.empty((len(LEAD_HOURS), size))
            for k, j in enumerate(lead_index):
                mean_b[k] = new_b[var][j].astype(np.float64).mean(axis=-1)
            record.mean_b[var] = mean_b
        new_c: dict[str, np.ndarray] = {}
        for var in VARIABLES:
            out_c = new_b[var].copy()
            for k, j in enumerate(lead_index):
                sc = shifts_c[var][k]
                moved = sc != 0.0
                if moved.any():
                    out_c[j, moved] = (new_b[var][j, moved].astype(np.float64) + sc[moved, None]).astype(
                        np.float32
                    )
            new_c[var] = out_c
        for j in lead_index:
            new_c["td2m"][j] = np.minimum(new_c["td2m"][j], new_c["t2m"][j])
        for var in VARIABLES:
            b[var] = a[var].copy(data=new_b[var][None])
            c2[var] = a[var].copy(data=new_c[var][None])
        self.history.append(record)
        defined = np.isfinite(record.citizen["t2m"][1]).astype(int)  # 3 h lead
        self.citizen_defined_count = (
            defined if self.citizen_defined_count is None else self.citizen_defined_count + defined
        )
        return b, c2, diag


# ----------------------------------------------------------------------------- Track B (5-fold LOAO)


def assign_spatial_folds(station_ids, lat, lon, n_folds: int = N_FOLDS) -> np.ndarray:
    """Deterministic greedy maximally-separated balanced folds from static coordinates only."""
    ids = np.asarray(station_ids, dtype=str)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    n = len(ids)
    if n_folds < 2 or n < n_folds:
        raise ValueError(f"need at least n_folds={n_folds} stations (got {n})")
    order = np.argsort(ids, kind="stable")
    caps = [n // n_folds + (1 if f < n % n_folds else 0) for f in range(n_folds)]
    members: list[list[int]] = [[] for _ in range(n_folds)]
    folds = np.full(n, -1, dtype=int)
    for rank, idx in enumerate(order):
        if rank < n_folds:
            folds[idx] = rank
            members[rank].append(int(idx))
            continue
        best_f = -1
        best_dist = -1.0
        for f in range(n_folds):
            if len(members[f]) >= caps[f]:
                continue
            m = np.asarray(members[f], dtype=int)
            dmin = float(distances_km(float(lat[idx]), float(lon[idx]), lat[m], lon[m]).min())
            if dmin > best_dist:
                best_dist = dmin
                best_f = f
        folds[idx] = best_f
        members[best_f].append(int(idx))
    return folds


def make_g_base(gefs_t: xr.Dataset, gefs_prev: xr.Dataset | None) -> xr.Dataset:
    """Construct Track B 6-lead base ensemble from raw GEFS (zero explicit ISD inputs).

    Lead 1 h is ``(2/3) * G(T - 12 h, 12 h) + (1/3) * G(T, 3 h)`` per member (falling back to
    ``G(T, 3 h)`` on January 1 00Z when no same-year prior cycle exists). Leads 3–48 h are exact
    ``gefs_t`` bytes.
    """
    if gefs_t.sizes["init_time"] != 1:
        raise ValueError("expected a single GEFS issue")
    lead3 = np.timedelta64(3, "h")
    lead12 = np.timedelta64(12, "h")
    lead1 = np.timedelta64(1, "h")
    leads_new = np.concatenate([[lead1], gefs_t.lead_time.values])
    data_vars = {}
    for var in gefs_t.data_vars:
        arr = gefs_t[var].transpose("init_time", "lead_time", "station_id", "ens_member").values
        v3 = gefs_t[var].sel(lead_time=lead3).transpose("init_time", "station_id", "ens_member").values
        if gefs_prev is not None:
            v12 = (
                gefs_prev[var].sel(lead_time=lead12).transpose("init_time", "station_id", "ens_member").values
            )
            v1 = ((2.0 / 3.0) * v12.astype(np.float64) + (1.0 / 3.0) * v3.astype(np.float64)).astype(
                np.float32
            )
        else:
            v1 = v3.astype(np.float32, copy=True)
        merged = np.concatenate([v1[:, None, :, :], arr], axis=1)
        data_vars[var] = (
            ("init_time", "lead_time", "station_id", "ens_member"),
            merged,
            dict(gefs_t[var].attrs),
        )
    if "td2m" in data_vars and "t2m" in data_vars:
        td = data_vars["td2m"][1].copy()
        td[:, 0] = np.minimum(td[:, 0], data_vars["t2m"][1][:, 0])
        data_vars["td2m"] = (data_vars["td2m"][0], td, data_vars["td2m"][2])
    coords = {k: v for k, v in gefs_t.coords.items() if k != "lead_time"}
    coords["lead_time"] = leads_new
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(gefs_t.attrs))


def _spread_gain(errors: np.ndarray, variances: np.ndarray, station_count: int) -> float:
    """v3 pooled spread-skill gain with prior g_0 = 1 and shrinkage ``station_count``."""
    n = errors.size
    if not n:
        return 1.0
    a_sum = float(np.cumsum(np.square(errors), dtype=np.float64)[-1])
    b_sum = float(np.cumsum(variances, dtype=np.float64)[-1])
    if b_sum <= 0.0:
        return 1.0
    return float(np.sqrt((n * (a_sum / b_sum) + station_count * 1.0) / (n + station_count)))


@dataclass
class _FoldIssue:
    init: np.datetime64
    mean_g_in: dict  # var -> (n_leads, S_in)
    var_g_in: dict  # var -> (n_leads, S_in) corrected variance ((M+1)/M)*var(ddof=1)
    mean_g_bias_in: dict  # var -> (n_leads, S_in)
    mean_b_in: dict  # var -> (n_leads, S_in) post-cap B_LOAO mean as issued
    off_in: dict  # var -> (n_leads, S_in)
    cit_in: dict  # var -> (n_leads, S_in)
    err12_in: dict  # var -> (S_in,) O(j, T) - G(j, T-12h, 12h)
    diff_cit: dict  # var -> (n_pairs,) citizen minus adjusted prior GEFS


@dataclass
class TrackBFoldDiagnostics:
    init: str
    fold: int
    g_loao: dict = field(default_factory=dict)
    phi_off: dict = field(default_factory=dict)
    gamma_loao: dict = field(default_factory=dict)
    off_defined_out: dict = field(default_factory=dict)
    citizen_defined_out: dict = field(default_factory=dict)


class TrackBFoldReplay:
    """One outer fold of Track B. ``truth_in_fold`` contains ONLY in-fold stations (``folds != fold_id``)."""

    def __init__(
        self,
        year: int,
        fold_id: int,
        folds: np.ndarray,
        truth_in_fold: xr.Dataset,
        full_station_ids: np.ndarray,
        full_lat: np.ndarray,
        full_lon: np.ndarray,
        full_height: np.ndarray,
        pairs_loao: OrientedPairs,
        c_height: np.ndarray,
        *,
        token=None,
    ):
        self.year = year
        self.fold_id = int(fold_id)
        self.start, self.end = v2.year_bounds(year)
        if truth_in_fold.valid_time.values.min() < self.start:
            raise ValueError("truth before the start of the year must not be passed in")
        self.folds = np.asarray(folds, dtype=int)
        self.in_idx = np.flatnonzero(self.folds != self.fold_id)
        self.out_idx = np.flatnonzero(self.folds == self.fold_id)
        self.full_ids = np.asarray(full_station_ids, dtype=str)
        expected_in_ids = self.full_ids[self.in_idx].tolist()
        if truth_in_fold.station_id.values.astype(str).tolist() != expected_in_ids:
            raise ValueError("truth_in_fold must contain exclusively the in-fold stations in order")
        self.truth_in = truth_in_fold
        self.n_all = len(self.full_ids)
        self.n_in = len(self.in_idx)
        self.full_height = np.asarray(full_height, dtype=np.float64)
        self.c_height = np.asarray(c_height, dtype=np.float64)
        self.pairs = pairs_loao
        self.token = None if token is None else np.unique(np.asarray(token), return_inverse=True)[1]
        # Precompute official-neighbour geometry: every target k (0..n_all-1) -> in-fold j' != k <= 150 km.
        lat_all = np.asarray(full_lat, dtype=np.float64)
        lon_all = np.asarray(full_lon, dtype=np.float64)
        lat_in, lon_in = lat_all[self.in_idx], lon_all[self.in_idx]
        self.off_nbrs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for k in range(self.n_all):
            rx, ry, km = haversine_displacement_km(float(lat_all[k]), float(lon_all[k]), lat_in, lon_in)
            ok = np.flatnonzero((km <= OFFICIAL_RADIUS_KM) & (self.in_idx != k))
            dist_w = np.exp(-np.square(km[ok] / OFFICIAL_DISTANCE_SCALE_KM) / 2.0)
            self.off_nbrs.append((ok, km[ok], rx[ok], ry[ok], dist_w))
        order = np.argsort(pairs_loao.official, kind="stable")
        self.by_official = (
            np.split(order, np.flatnonzero(np.diff(pairs_loao.official[order])) + 1) if order.size else []
        )
        # Precompute lapse adjustment per (citizen, official) pair: +6.5 * (h_k - h_c) / 1000
        self.pair_lapse = (
            LAPSE_K_PER_KM
            * (self.full_height[pairs_loao.official] - self.c_height[pairs_loao.citizen])
            / 1000.0
            if pairs_loao.citizen.size
            else np.empty(0, dtype=np.float64)
        )
        self.history: deque[_FoldIssue] = deque()

    def _reading_in(self, var: str, t) -> np.ndarray:
        if t not in self.truth_in.indexes["valid_time"]:
            return np.full(self.n_in, np.nan)
        slot = self.truth_in.sel(valid_time=t)
        value = slot[var].values.astype(np.float64)
        offset = slot[var + "_offset_min"].values.astype(np.float64)
        ok = np.isfinite(value) & (offset >= -MAX_OFFSET_MIN) & (offset <= 0)
        ok &= offset >= (self.start - t) / MINUTE
        return np.where(ok, value, np.nan)

    def _verified_in(self, var: str, t, lead_index: int, lead):
        past = [
            item
            for item in self.history
            if item.init < t and (t - item.init) % DAY == np.timedelta64(0, "ns")
        ]
        past = [item for item in past if item.init + lead + MAX_OFFSET_MIN * MINUTE > t - GAIN_WINDOW]
        if not past:
            return None
        labels = np.array([item.init + lead for item in past], dtype="datetime64[ns]")
        slots = self.truth_in.sel(valid_time=labels)
        y = slots[var].values.astype(np.float64)
        offsets = slots[var + "_offset_min"].values.astype(np.float64)
        age_min = ((t - labels) / MINUTE)[:, None] - offsets
        ok = (
            np.isfinite(y)
            & (np.abs(offsets) <= MAX_OFFSET_MIN)
            & (age_min >= 0)
            & (age_min < GAIN_WINDOW / MINUTE)
            & (age_min <= (t - self.start) / MINUTE)
        )
        return past, ok, y

    def _bias_off(self, var: str, t, lead_index: int, lead) -> np.ndarray:
        """Regional GEFS bias per station k from in-fold neighbours j' != k within 150 km."""
        bias = np.zeros(self.n_all, dtype=np.float64)
        ver = self._verified_in(var, t, lead_index, lead)
        if ver is None:
            return bias
        past, ok, y = ver
        err_g = np.where(ok, y - np.stack([item.mean_g_in[var][lead_index] for item in past]), np.nan)
        count = np.isfinite(err_g).sum(axis=0)
        med = np.full(self.n_in, np.nan)
        enough = count >= USUAL_MIN
        if enough.any():
            med[enough] = np.nanmedian(err_g[:, enough], axis=0)
        for k, (nbr_in, _, _, _, dist_w) in enumerate(self.off_nbrs):
            if not nbr_in.size:
                continue
            vals = med[nbr_in]
            good = np.isfinite(vals)
            if good.any():
                bias[k] = arms.weighted_lower_median(vals[good], dist_w[good])
        return bias

    def _off_loao(self, var: str, t, reading_in, gefs_prev_mean, u10, v10):
        n_leads = len(LEAD_HOURS)
        out = np.full((n_leads, self.n_all), np.nan)
        if gefs_prev_mean is None:
            return out, np.full(self.n_in, np.nan)
        g_in = gefs_prev_mean[var][self.in_idx]
        err12_in = np.where(np.isfinite(reading_in) & np.isfinite(g_in), reading_in - g_in, np.nan)
        earlier = [
            item.err12_in[var]
            for item in self.history
            if t - USUAL_DAYS * DAY <= item.init <= t - DAY
            and (t - item.init) % DAY == np.timedelta64(0, "ns")
        ]
        if not earlier:
            return out, err12_in
        stack = np.stack(earlier)
        count = np.isfinite(stack).sum(axis=0)
        usual = np.full(self.n_in, np.nan)
        enough = count >= USUAL_MIN
        if enough.any():
            usual[enough] = np.nanmedian(stack[:, enough], axis=0)
        dep = np.clip(err12_in - usual, -CLIP_K, CLIP_K)
        for k, (nbr_in, km, rx, ry, _) in enumerate(self.off_nbrs):
            if not nbr_in.size:
                continue
            vals = dep[nbr_in]
            good = np.isfinite(vals)
            if int(good.sum()) < MIN_OFFICIAL_NEIGHBOURS:
                continue
            for li, lead_h in enumerate(LEAD_HOURS):
                px, py = upwind_target_km(float(u10[k]), float(v10[k]), lead_h, OFFICIAL_RADIUS_KM)
                w = upwind_gaussian_weight(rx[good], ry[good], km[good], px, py, OFFICIAL_DISTANCE_SCALE_KM)
                out[li, k] = arms.weighted_lower_median(vals[good], w)
        return out, err12_in

    def _citizen_loao(self, var: str, t, gefs_prev_mean, value, weight, age, u10, v10):
        p = self.pairs
        n_leads = len(LEAD_HOURS)
        out = np.full((n_leads, self.n_all), np.nan)
        if not p.citizen.size or gefs_prev_mean is None:
            return out, np.full(p.citizen.size, np.nan)
        g_adj = gefs_prev_mean[var][p.official] + self.pair_lapse
        y, w = value[p.citizen], weight[p.citizen]
        with np.errstate(invalid="ignore"):
            eligible = np.isfinite(y) & (w > 0.0) & np.isfinite(g_adj)
        diff = np.where(eligible, y - g_adj, np.nan)
        earlier = [
            item.diff_cit[var]
            for item in self.history
            if t - USUAL_DAYS * DAY <= item.init <= t - DAY
            and (t - item.init) % DAY == np.timedelta64(0, "ns")
        ]
        if not earlier:
            return out, diff
        stack = np.stack(earlier)
        count = np.isfinite(stack).sum(axis=0)
        usual = np.full(p.citizen.size, np.nan)
        enough = count >= USUAL_MIN
        if enough.any():
            usual[enough] = np.nanmedian(stack[:, enough], axis=0)
        dep = np.clip(diff - usual, -CLIP_K, CLIP_K)
        w_eff = temper_trust_weight(w)
        for rows in self.by_official:
            use_base = rows[np.isfinite(dep[rows])]
            if use_base.size < MIN_NEIGHBOURS:
                continue
            k = int(p.official[use_base[0]])
            for li, lead_h in enumerate(LEAD_HOURS):
                px, py = upwind_target_km(float(u10[k]), float(v10[k]), lead_h, RADIUS_KM)
                geo_w = upwind_gaussian_weight(
                    p.rx_km[use_base], p.ry_km[use_base], p.distance_km[use_base], px, py, DISTANCE_SCALE_KM
                )
                weights = w_eff[use_base] * geo_w
                keep = _one_per_token(self.token, p.citizen, use_base, age, weights)
                use, weights = use_base[keep], weights[keep]
                if use.size < MIN_NEIGHBOURS or np.unique(p.cell[use]).size < MIN_CELLS:
                    continue
                out[li, k] = arms.weighted_lower_median(dep[use], weights)
        return out, diff

    def issue(
        self,
        g_base: xr.Dataset,
        gefs_prev_mean: dict[str, np.ndarray] | None,
        snapshot: dict,
    ) -> tuple[xr.Dataset, xr.Dataset, TrackBFoldDiagnostics]:
        """Step one issue for this outer fold. Returns full-grid B_LOAO, C2_LOAO (held-out slices
        belong to ``self.out_idx``), and fold diagnostics."""
        t = g_base.init_time.values[0]
        if not self.start <= t or t + v2.LEADS[-1] >= self.end:
            raise ValueError("issue outside the year")
        if self.history and t <= self.history[-1].init:
            raise ValueError("initializations must increase")
        if g_base.station_id.values.astype(str).tolist() != self.full_ids.tolist():
            raise ValueError("g_base station grid does not match fold station grid")
        while self.history and self.history[0].init < t - (USUAL_DAYS + 2) * DAY:
            self.history.popleft()
        if gefs_prev_mean is None:
            u10 = np.zeros(self.n_all, dtype=np.float64)
            v10 = np.zeros(self.n_all, dtype=np.float64)
        else:
            u10 = np.asarray(gefs_prev_mean["u10"], dtype=np.float64)
            v10 = np.asarray(gefs_prev_mean["v10"], dtype=np.float64)
        lead_values = g_base.lead_time.values
        lead_index = [int(np.flatnonzero(lead_values == h * HOUR)[0]) for h in LEAD_HOURS]
        n_members = g_base.sizes["ens_member"]
        var_factor = (n_members + 1.0) / n_members
        diag = TrackBFoldDiagnostics(init=str(np.datetime_as_string(t, unit="h")), fold=self.fold_id)
        rec = _FoldIssue(t, {}, {}, {}, {}, {}, {}, {}, {})
        b_ds, c_ds = g_base.copy(deep=False), g_base.copy(deep=False)
        b_ds.attrs = {**g_base.attrs, "model_id": MODEL_LOAO_BASE, "model_version": VERSION}
        c_ds.attrs = {**g_base.attrs, "model_id": MODEL_LOAO_CITIZEN, "model_version": VERSION}
        new_b: dict[str, np.ndarray] = {}
        shifts_c: dict[str, list[np.ndarray]] = {}
        for var in VARIABLES:
            members = g_base[var].values[0].astype(np.float64)  # (lead, station, member)
            mean_g = members.mean(axis=-1)
            var_g = var_factor * members.var(axis=-1, ddof=1)
            reading_in = self._reading_in(var, t)
            off_all, err12_in = self._off_loao(var, t, reading_in, gefs_prev_mean, u10, v10)
            value, weight, *rest = snapshot[var]
            value, weight = np.asarray(value, float), np.asarray(weight, float)
            age = np.asarray(rest[0]).astype(np.int64) if rest else np.zeros(len(value), dtype=np.int64)
            cit_all, diff_cit = self._citizen_loao(var, t, gefs_prev_mean, value, weight, age, u10, v10)
            out_b = members.copy()
            mean_g_bias_in = np.empty((len(LEAD_HOURS), self.n_in))
            diag.g_loao[var], diag.phi_off[var], diag.gamma_loao[var] = [], [], []
            shifts_c[var] = []
            for k, (j, hours) in enumerate(zip(lead_index, LEAD_HOURS, strict=True)):
                bias_off = self._bias_off(var, t, k, hours * HOUR)
                g_gain, phi_off, gamma_loao = 1.0, 0.0, 0.0
                ver = self._verified_in(var, t, k, hours * HOUR)
                if ver is not None:
                    past, ok, y_in = ver
                    err_gb = y_in - np.stack([item.mean_g_bias_in[var][k] for item in past])
                    v_past = np.stack([item.var_g_in[var][k] for item in past])
                    err_b = y_in - np.stack([item.mean_b_in[var][k] for item in past])
                    off_p = np.stack([item.off_in[var][k] for item in past])
                    cit_p = np.stack([item.cit_in[var][k] for item in past])
                    use_g = ok & np.isfinite(err_gb) & np.isfinite(v_past)
                    g_gain = _spread_gain(err_gb[use_g], v_past[use_g], self.n_in)
                    use_o = ok & np.isfinite(off_p)
                    phi_off = arms.gain(off_p[use_o], err_gb[use_o], self.n_in)
                    use_c = ok & np.isfinite(cit_p)
                    gamma_loao = arms.gain(cit_p[use_c], err_b[use_c], self.n_in)
                centre_gb = mean_g[j] + bias_off
                mean_g_bias_in[k] = centre_gb[self.in_idx]
                shift_off = np.where(np.isfinite(off_all[k]), phi_off * off_all[k], 0.0)
                centre_b = centre_gb + shift_off
                b_mem = centre_b[:, None] + g_gain * (members[j] - mean_g[j][:, None])
                shift_cit = np.where(np.isfinite(cit_all[k]), gamma_loao * cit_all[k], 0.0)
                out_b[j] = b_mem
                shifts_c[var].append(shift_cit)
                diag.g_loao[var].append(g_gain)
                diag.phi_off[var].append(phi_off)
                diag.gamma_loao[var].append(gamma_loao)
            new_b[var] = out_b.astype(np.float32)
            rec.mean_g_in[var] = mean_g[lead_index][:, self.in_idx]
            rec.var_g_in[var] = var_g[lead_index][:, self.in_idx]
            rec.mean_g_bias_in[var] = mean_g_bias_in
            rec.off_in[var] = off_all[:, self.in_idx]
            rec.cit_in[var] = cit_all[:, self.in_idx]
            rec.err12_in[var] = err12_in
            rec.diff_cit[var] = diff_cit
            diag.off_defined_out[var] = int(np.isfinite(off_all[1, self.out_idx]).sum())
            diag.citizen_defined_out[var] = int(np.isfinite(cit_all[1, self.out_idx]).sum())
            if var == "t2m":
                self.last_cit_defined_out_t2m_3h = np.isfinite(cit_all[1, self.out_idx]).astype(int)
        # Finish control B_LOAO first (including td2m <= t2m cap), then add citizen shifts to finished B_LOAO.
        for j in lead_index:
            new_b["td2m"][j] = np.minimum(new_b["td2m"][j], new_b["t2m"][j])
        for var in VARIABLES:
            mean_b_in = np.empty((len(LEAD_HOURS), self.n_in))
            for k, j in enumerate(lead_index):
                mean_b_in[k] = new_b[var][j, self.in_idx].astype(np.float64).mean(axis=-1)
            rec.mean_b_in[var] = mean_b_in
        new_c: dict[str, np.ndarray] = {}
        for var in VARIABLES:
            out_c = new_b[var].copy()
            for k, j in enumerate(lead_index):
                sc = shifts_c[var][k]
                moved = sc != 0.0
                if moved.any():
                    out_c[j, moved] = (new_b[var][j, moved].astype(np.float64) + sc[moved, None]).astype(
                        np.float32
                    )
            new_c[var] = out_c
        for j in lead_index:
            new_c["td2m"][j] = np.minimum(new_c["td2m"][j], new_c["t2m"][j])
        for var in VARIABLES:
            b_ds[var] = g_base[var].copy(data=new_b[var][None])
            c_ds[var] = g_base[var].copy(data=new_c[var][None])
        self.history.append(rec)
        return b_ds, c_ds, diag


class TrackBReplay:
    """Assembles full-grid held-out ``B_LOAO`` and ``C2_LOAO`` across all 5 outer folds."""

    def __init__(
        self,
        year: int,
        truth: xr.Dataset,
        folds: np.ndarray,
        pairs_loao: OrientedPairs,
        c_height: np.ndarray,
        *,
        token=None,
    ):
        self.year = year
        self.folds = np.asarray(folds, dtype=int)
        self.n_folds = int(self.folds.max()) + 1
        full_ids = truth.station_id.values.astype(str)
        full_lat = truth.latitude.values.astype(np.float64)
        full_lon = truth.longitude.values.astype(np.float64)
        full_height = truth.elevation_m.values.astype(np.float64)
        self.fold_replays = []
        for f in range(self.n_folds):
            in_idx = np.flatnonzero(self.folds != f)
            truth_in = truth.isel(station_id=in_idx).copy(deep=True)
            self.fold_replays.append(
                TrackBFoldReplay(
                    year,
                    f,
                    self.folds,
                    truth_in,
                    full_ids,
                    full_lat,
                    full_lon,
                    full_height,
                    pairs_loao,
                    c_height,
                    token=token,
                )
            )
        self.citizen_defined_count = np.zeros(len(full_ids), dtype=int)

    def issue(
        self,
        g_base: xr.Dataset,
        gefs_prev_mean: dict[str, np.ndarray] | None,
        snapshot: dict,
    ) -> tuple[xr.Dataset, xr.Dataset, list[TrackBFoldDiagnostics]]:
        """Step all folds and stitch each station exclusively from the fold holding it out."""
        b_full = g_base.copy(deep=True)
        c_full = g_base.copy(deep=True)
        b_full.attrs = {**g_base.attrs, "model_id": MODEL_LOAO_BASE, "model_version": VERSION}
        c_full.attrs = {**g_base.attrs, "model_id": MODEL_LOAO_CITIZEN, "model_version": VERSION}
        diags = []
        for replay in self.fold_replays:
            b_f, c_f, diag = replay.issue(g_base, gefs_prev_mean, snapshot)
            diags.append(diag)
            out = replay.out_idx
            for var in VARIABLES:
                b_full[var].values[:, :, out, :] = b_f[var].values[:, :, out, :]
                c_full[var].values[:, :, out, :] = c_f[var].values[:, :, out, :]
            self.citizen_defined_count[out] += replay.last_cit_defined_out_t2m_3h
        return b_full, c_full, diags


def track_b_covered_counts(
    year: int,
    issues: np.ndarray,
    full_lat: np.ndarray,
    full_lon: np.ndarray,
    full_height: np.ndarray,
    pairs_loao: OrientedPairs,
    c_height: np.ndarray,
    gefs_prev_means: list[dict[str, np.ndarray] | None],
    snapshots: list[dict],
    *,
    token=None,
) -> np.ndarray:
    """Compute per-station count of issues where ``citizen_loao(s, T, 3 h)`` for ``t2m`` is defined.

    Depends strictly on the official-free citizen extract, static coordinates, and prior GEFS
    means — zero ISD official observations enter coverage determination.
    """
    tok = None if token is None else np.unique(np.asarray(token), return_inverse=True)[1]
    n_all = len(full_lat)
    order = np.argsort(pairs_loao.official, kind="stable")
    by_official = (
        np.split(order, np.flatnonzero(np.diff(pairs_loao.official[order])) + 1) if order.size else []
    )
    pair_lapse = (
        LAPSE_K_PER_KM * (full_height[pairs_loao.official] - c_height[pairs_loao.citizen]) / 1000.0
        if pairs_loao.citizen.size
        else np.empty(0, dtype=np.float64)
    )
    history: deque[tuple[np.datetime64, np.ndarray]] = deque()
    counts = np.zeros(n_all, dtype=int)
    for t, prev_mean, snap in zip(issues, gefs_prev_means, snapshots, strict=True):
        while history and history[0][0] < t - (USUAL_DAYS + 2) * DAY:
            history.popleft()
        if not pairs_loao.citizen.size or prev_mean is None:
            history.append((t, np.full(pairs_loao.citizen.size, np.nan)))
            continue
        value, weight, *rest = snap["t2m"]
        value, weight = np.asarray(value, float), np.asarray(weight, float)
        age = np.asarray(rest[0]).astype(np.int64) if rest else np.zeros(len(value), dtype=np.int64)
        g_adj = prev_mean["t2m"][pairs_loao.official] + pair_lapse
        y, w = value[pairs_loao.citizen], weight[pairs_loao.citizen]
        with np.errstate(invalid="ignore"):
            eligible = np.isfinite(y) & (w > 0.0) & np.isfinite(g_adj)
        diff = np.where(eligible, y - g_adj, np.nan)
        earlier = [
            d
            for it, d in history
            if t - USUAL_DAYS * DAY <= it <= t - DAY and (t - it) % DAY == np.timedelta64(0, "ns")
        ]
        if earlier:
            stack = np.stack(earlier)
            cnt = np.isfinite(stack).sum(axis=0)
            usual = np.full(pairs_loao.citizen.size, np.nan)
            enough = cnt >= USUAL_MIN
            if enough.any():
                usual[enough] = np.nanmedian(stack[:, enough], axis=0)
            dep = np.clip(diff - usual, -CLIP_K, CLIP_K)
            w_eff = temper_trust_weight(w)
            u10, v10 = np.asarray(prev_mean["u10"], float), np.asarray(prev_mean["v10"], float)
            for rows in by_official:
                use_base = rows[np.isfinite(dep[rows])]
                if use_base.size < MIN_NEIGHBOURS:
                    continue
                k = int(pairs_loao.official[use_base[0]])
                px, py = upwind_target_km(float(u10[k]), float(v10[k]), 3.0, RADIUS_KM)
                geo_w = upwind_gaussian_weight(
                    pairs_loao.rx_km[use_base],
                    pairs_loao.ry_km[use_base],
                    pairs_loao.distance_km[use_base],
                    px,
                    py,
                    DISTANCE_SCALE_KM,
                )
                weights = w_eff[use_base] * geo_w
                keep = _one_per_token(tok, pairs_loao.citizen, use_base, age, weights)
                use = use_base[keep]
                if use.size >= MIN_NEIGHBOURS and np.unique(pairs_loao.cell[use]).size >= MIN_CELLS:
                    counts[k] += 1
        history.append((t, diff))
    return counts


# ----------------------------------------------------------------------------- post-freeze relay audit


def target_paired_relay_audit(
    contributing: list[tuple[int, int, str, int, np.datetime64, float]],
    citizen_streams: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    official_series: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    covered_stations_idx: set[int],
    trial,
    *,
    standard_ever_copycat: np.ndarray | None = None,
) -> dict:
    """Post-freeze aggregate relay diagnostics for Track B (zero identifiers or coordinates).

    ``contributing`` entries are ``(citizen_idx, official_idx, var, lead_h, issue_time, weight)``
    emitted strictly after ``C2_LOAO`` and ``COVERED_LOAO`` are frozen.

    ``citizen_streams[(citizen_idx, var)]`` holds the retained full-resolution ``(times_ns, values)``
    citizen stream from ``<year>-tiles/`` (never the twice-daily extract). Each contributing
    ``(citizen, target_s, var)`` triple is paired against ``target_s`` using frozen
    ``trial.official_evidence`` (as-of <= 15 min, strictly one pair per distinct official measurement
    timestamp, trailing 7-day window through ``T``) and evaluated with ``suspected_copy``.
    """
    cfg, sigma = trial.CFG, trial.SIGMA
    # Cache target-paired copycat flags per (citizen_idx, official_idx, var) -> (times_ns, is_copy)
    cache: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    totals = {
        s_name: {
            v: {"total_weight": 0.0, "target_flagged_weight": 0.0, "std_ref_flagged_weight": 0.0}
            for v in VARIABLES
        }
        for s_name in ("ALL", "COVERED_LOAO")
    }
    for c_idx, s_idx, var, _lead_h, issue_t, w in contributing:
        if w <= 0.0 or not np.isfinite(w):
            continue
        key = (int(c_idx), int(s_idx), str(var))
        if key not in cache:
            stream = citizen_streams.get((int(c_idx), str(var)))
            if stream is None or len(stream[0]) == 0:
                cache[key] = (np.empty(0, dtype="datetime64[ns]"), np.empty(0, dtype=bool))
            else:
                when, vals = stream
                _, _, r, rms, count, span = trial.official_evidence(
                    official_series[var][int(s_idx)], when, vals
                )
                flagged = suspected_copy(r, rms, sigma, cfg, thermal=True, count=count, hours=span)
                cache[key] = (when, flagged.astype(bool))
        when, flagged = cache[key]
        is_target_copy = False
        if len(when):
            pos = int(np.searchsorted(when, np.datetime64(issue_t, "ns"), side="right")) - 1
            if pos >= 0:
                is_target_copy = bool(flagged[pos])
        is_std_copy = (
            bool(standard_ever_copycat[int(c_idx)])
            if standard_ever_copycat is not None and int(c_idx) < len(standard_ever_copycat)
            else False
        )
        sets = ("ALL", "COVERED_LOAO") if int(s_idx) in covered_stations_idx else ("ALL",)
        for s_name in sets:
            bucket = totals[s_name][var]
            bucket["total_weight"] += float(w)
            if is_target_copy:
                bucket["target_flagged_weight"] += float(w)
            if is_std_copy:
                bucket["std_ref_flagged_weight"] += float(w)
    out = {}
    for s_name, per_var in totals.items():
        out[s_name] = {}
        for var, b in per_var.items():
            denom = b["total_weight"]
            out[s_name][var] = {
                "total_weight": denom,
                "target_paired_relay_share": (
                    b["target_flagged_weight"] / denom if denom > 0.0 else "unavailable"
                ),
                "standard_qc_nearest_ref_ever_flagged_share": (
                    (b["std_ref_flagged_weight"] / denom if denom > 0.0 else "unavailable")
                    if standard_ever_copycat is not None
                    else "unavailable"
                ),
            }
    return out
