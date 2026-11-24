"""Phase 3 arms B and C, exactly as registered in ``docs/PHASE3_DESIGN.md`` section 1.2.

A is the frozen ``gefs-station-corrected`` version 3 store. B adds the airport's own reading
(official data only). C adds the QC-weighted citizen neighbourhood on top of B. Only ``t2m`` and
``td2m`` at leads 1, 3 and 6 h are changed; every other value is the v3 value, byte for byte.

Nothing measured after the issue time is read, and no state crosses a year boundary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from wrfcomp.assimilation.qc import cells, distances_km
from wrfcomp.models import gefs_station_v2 as v2

MODEL_OWN = "gefs-station-corrected-own"
MODEL_CITIZEN = "gefs-station-corrected-citizen"
VERSION = "1"
VARIABLES = ("t2m", "td2m")
LEAD_HOURS = (1, 3, 6)
OWN_LEAD_HOURS = 12  # the earlier v3 forecast that verifies exactly at the issue time
RADIUS_KM = 50.0
DISTANCE_SCALE_KM = 25.0
MAX_HEIGHT_DIFFERENCE_M = 200.0
USUAL_DAYS = 30
USUAL_MIN = 10
CLIP_K = 5.0
MIN_NEIGHBOURS = 3
MIN_CELLS = 2
GAIN_WINDOW = np.timedelta64(30, "D")
MAX_OFFSET_MIN = 15
DAY = np.timedelta64(1, "D")
HOUR = np.timedelta64(1, "h")
MINUTE = np.timedelta64(1, "m")


@dataclass(frozen=True)
class Pairs:
    """Every (citizen station, official station) pair that rule 4 allows, in a fixed order."""

    citizen: np.ndarray
    official: np.ndarray
    distance_km: np.ndarray
    cell: np.ndarray  # tenth-degree cell of the citizen station


def neighbour_pairs(c_lat, c_lon, c_height, o_lat, o_lon, o_height) -> Pairs:
    cit, off, dist = [], [], []
    c_lat, c_lon, c_height = (np.asarray(a, dtype=np.float64) for a in (c_lat, c_lon, c_height))
    for s in range(len(o_lat)):
        km = distances_km(float(o_lat[s]), float(o_lon[s]), c_lat, c_lon)
        ok = np.flatnonzero(
            (km <= RADIUS_KM) & (np.abs(c_height - float(o_height[s])) <= MAX_HEIGHT_DIFFERENCE_M)
        )
        cit.append(ok)
        off.append(np.full(ok.size, s))
        dist.append(km[ok])
    citizen = np.concatenate(cit).astype(int) if cit else np.empty(0, int)
    return Pairs(
        citizen=citizen,
        official=np.concatenate(off).astype(int) if off else np.empty(0, int),
        distance_km=np.concatenate(dist) if dist else np.empty(0),
        cell=cells(c_lat[citizen], c_lon[citizen]) if citizen.size else np.empty(0, int),
    )


def weighted_lower_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Smallest value whose cumulative weight reaches half the total. Ties keep input order."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2, side="left")])


def gain(predictor: np.ndarray, error: np.ndarray, station_count: int) -> float:
    """Rule 8. Inputs in ascending issue and station order; one entry per eligible record."""
    n = predictor.size
    if not n:
        return 0.0
    denominator = np.cumsum(np.square(predictor), dtype=np.float64)[-1]
    if denominator == 0:
        return 0.0
    slope = np.cumsum(predictor * error, dtype=np.float64)[-1] / denominator
    return float(np.clip(n / (n + station_count) * slope, 0.0, 1.0))


@dataclass
class _Issue:
    init: np.datetime64
    own: dict  # variable -> (station,) own innovation or NaN
    citizen: dict  # variable -> (station,) neighbourhood value or NaN
    mean_a: dict  # variable -> (lead, station) v3 member mean
    mean_b: dict  # variable -> (lead, station) B member mean as issued
    mean_own_lead: dict  # variable -> (station,) v3 member mean at lead 12 h
    difference: dict  # variable -> (pair,) citizen minus airport, or NaN


@dataclass
class Diagnostics:
    init: str
    phi: dict = field(default_factory=dict)  # variable -> [gain per lead]
    gamma: dict = field(default_factory=dict)
    own_defined: dict = field(default_factory=dict)  # variable -> stations
    citizen_defined: dict = field(default_factory=dict)
    weight_total: dict = field(default_factory=dict)  # neighbourhood weight used
    weight_flagged: dict = field(default_factory=dict)  # of which stations ever flagged as copies


class Replay:
    """One ordered same-year replay of both arms."""

    def __init__(self, year: int, truth: xr.Dataset, pairs: Pairs, flagged=None, token=None):
        """``token``: one label per citizen station row. Rows that share a label are one physical
        station stored at several positions; they count once (see ``_one_per_token``)."""
        self.year = year
        self.start, self.end = v2.year_bounds(year)
        if truth.valid_time.values.min() < self.start:
            raise ValueError("truth before the start of the year must not be passed in")
        self.truth = truth
        self.pairs = pairs
        self.flagged = np.zeros(0, bool) if flagged is None else np.asarray(flagged, dtype=bool)
        self.token = None if token is None else np.unique(np.asarray(token), return_inverse=True)[1]
        self.pair_weight = np.exp(-np.square(pairs.distance_km / DISTANCE_SCALE_KM) / 2)
        order = np.argsort(pairs.official, kind="stable")
        self.by_official = (
            np.split(order, np.flatnonzero(np.diff(pairs.official[order])) + 1) if order.size else []
        )
        self.history: deque[_Issue] = deque()
        self.citizen_defined_count = None

    # -- inputs at the issue time ------------------------------------------------------------

    def _reading(self, var: str, t) -> np.ndarray:
        """Rule 1: the official value at label T, measured at or before T."""
        size = self.truth.sizes["station_id"]
        if t not in self.truth.indexes["valid_time"]:
            return np.full(size, np.nan)
        slot = self.truth.sel(valid_time=t)
        value = slot[var].values.astype(np.float64)
        offset = slot[var + "_offset_min"].values.astype(np.float64)
        ok = np.isfinite(value) & (offset >= -MAX_OFFSET_MIN) & (offset <= 0)
        # Year isolation on the measurement time: a 00Z label on 1 January with a negative offset
        # was measured in the previous year.
        ok &= offset >= (self.start - t) / MINUTE
        return np.where(ok, value, np.nan)

    def _one_per_token(self, use, age, weights):
        """Of several rows of one token, keep the one with the newest report (then the larger
        weight, then the lower row). A station counts once towards the three-station minimum and
        once in the median, whatever positions it is stored at."""
        if self.token is None or use.size < 2:
            return np.arange(use.size)
        token = self.token[self.pairs.citizen[use]]
        order = np.lexsort((np.arange(use.size), -weights, age[self.pairs.citizen[use]], token))
        first = np.r_[True, token[order][1:] != token[order][:-1]]
        return np.sort(order[first])

    def _neighbourhood(self, var, t, reading, value, weight, age, diag):
        """Rules 3 to 7 for every official station: the neighbourhood value and the pair differences."""
        p = self.pairs
        size = self.truth.sizes["station_id"]
        result = np.full(size, np.nan)
        if not p.citizen.size:
            return result, np.empty(0)
        y, w = value[p.citizen], weight[p.citizen]
        with np.errstate(invalid="ignore"):
            eligible = np.isfinite(y) & (w > 0) & np.isfinite(reading[p.official])
        difference = np.where(eligible, y - reading[p.official], np.nan)
        earlier = [
            item.difference[var]
            for item in self.history
            if t - USUAL_DAYS * DAY <= item.init <= t - DAY
            and (t - item.init) % DAY == np.timedelta64(0, "ns")
        ]
        if not earlier:
            return result, difference
        stack = np.stack(earlier)
        count = np.isfinite(stack).sum(axis=0)
        usual = np.full(p.citizen.size, np.nan)
        enough = count >= USUAL_MIN
        if enough.any():
            usual[enough] = np.nanmedian(stack[:, enough], axis=0)
        departure = np.clip(difference - usual, -CLIP_K, CLIP_K)
        total = flagged = 0.0
        for rows in self.by_official:
            use = rows[np.isfinite(departure[rows])]
            weights = w[use] * self.pair_weight[use]
            keep = self._one_per_token(use, age, weights)
            use, weights = use[keep], weights[keep]
            if use.size < MIN_NEIGHBOURS or np.unique(p.cell[use]).size < MIN_CELLS:
                continue
            result[p.official[use[0]]] = weighted_lower_median(departure[use], weights)
            total += float(weights.sum())
            if self.flagged.size:
                flagged += float(weights[self.flagged[p.citizen[use]]].sum())
        diag.weight_total[var], diag.weight_flagged[var] = total, flagged
        return result, difference

    def _records(self, var, t, lead_index, lead):
        """Rule 8: earlier same-hour issues whose forecast for this lead has verified by T."""
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
        citizen = np.stack([item.citizen[var] for item in past])
        return ok, error_a, error_b, own, citizen

    # -- one issue ---------------------------------------------------------------------------

    def issue(self, a: xr.Dataset, snapshot: dict):
        """``a`` is one v3 issue. ``snapshot[var]`` is (value, weight) or (value, weight, age in
        seconds) per citizen station row at this issue: the latest positive-weight report measured in
        the 15 minutes up to the issue time (NaN and 0 when there is none). Returns (B, C, diagnostics)."""
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
        lead_values = a.lead_time.values
        lead_index = [int(np.flatnonzero(lead_values == h * HOUR)[0]) for h in LEAD_HOURS]
        own_index = int(np.flatnonzero(lead_values == OWN_LEAD_HOURS * HOUR)[0])
        while self.history and self.history[0].init < t - (USUAL_DAYS + 2) * DAY:
            self.history.popleft()
        previous = next((item for item in self.history if item.init == t - OWN_LEAD_HOURS * HOUR), None)
        size = len(stations)
        diag = Diagnostics(init=str(np.datetime_as_string(t, unit="h")))
        record = _Issue(t, {}, {}, {}, {}, {}, {})
        b, c = a.copy(deep=False), a.copy(deep=False)
        b.attrs = {**a.attrs, "model_id": MODEL_OWN, "model_version": VERSION}
        c.attrs = {**a.attrs, "model_id": MODEL_CITIZEN, "model_version": VERSION}
        new = {}
        for var in VARIABLES:
            members = a[var].values[0].astype(np.float64)  # (lead, station, member)
            mean = members.mean(axis=-1)
            reading = self._reading(var, t)
            own = reading - previous.mean_own_lead[var] if previous is not None else np.full(size, np.nan)
            value, weight, *rest = snapshot[var]
            value, weight = np.asarray(value, float), np.asarray(weight, float)
            age = np.asarray(rest[0], float) if rest else np.zeros(len(value))
            citizen, difference = self._neighbourhood(var, t, reading, value, weight, age, diag)
            out_b, out_c = members.copy(), members.copy()
            mean_b = np.empty((len(LEAD_HOURS), size))
            diag.phi[var], diag.gamma[var] = [], []
            for k, (j, hours) in enumerate(zip(lead_index, LEAD_HOURS, strict=True)):
                phi = gamma = 0.0
                records = self._records(var, t, k, hours * HOUR)
                if records is not None:
                    ok, error_a, error_b, past_own, past_citizen = records
                    use = ok & np.isfinite(past_own)
                    phi = gain(past_own[use], error_a[use], size)
                    use = ok & np.isfinite(past_citizen)
                    gamma = gain(past_citizen[use], error_b[use], size)
                shift_b = np.where(np.isfinite(own), phi * own, 0.0)
                shift_c = shift_b + np.where(np.isfinite(citizen), gamma * citizen, 0.0)
                out_b[j] = members[j] + shift_b[:, None]
                out_c[j] = members[j] + shift_c[:, None]
                diag.phi[var].append(phi)
                diag.gamma[var].append(gamma)
            new[var] = (out_b.astype(np.float32), out_c.astype(np.float32))
            for k, j in enumerate(lead_index):
                mean_b[k] = new[var][0][j].astype(np.float64).mean(axis=-1)
            record.own[var], record.citizen[var], record.difference[var] = own, citizen, difference
            record.mean_a[var] = mean[lead_index]
            record.mean_b[var] = mean_b
            record.mean_own_lead[var] = mean[own_index]
            diag.own_defined[var] = int(np.isfinite(own).sum())
            diag.citizen_defined[var] = int(np.isfinite(citizen).sum())
        # Rule 9: dewpoint never above the paired temperature, at the changed leads only.
        for arm in (0, 1):
            td, t2 = new["td2m"][arm], new["t2m"][arm]
            for j in lead_index:
                td[j] = np.minimum(td[j], t2[j])
        # The B mean used later as "B as issued" is the published one, after the cap.
        for k, j in enumerate(lead_index):
            record.mean_b["td2m"][k] = new["td2m"][0][j].astype(np.float64).mean(axis=-1)
        for var in VARIABLES:
            b[var] = a[var].copy(data=new[var][0][None])
            c[var] = a[var].copy(data=new[var][1][None])
        self.history.append(record)
        defined = np.isfinite(record.citizen["t2m"]).astype(int)
        self.citizen_defined_count = (
            defined if self.citizen_defined_count is None else self.citizen_defined_count + defined
        )
        return b, c, diag


def covered_stations(station_ids, defined_count, n_issues: int) -> list[str]:
    """COVERED (section 2): the neighbourhood value for t2m was defined at half or more of the issues."""
    return [str(s) for s, n in zip(station_ids, defined_count, strict=True) if 2 * int(n) >= n_issues]
