# WRFCoin Proof-of-Forecast-Skill (PoFS) Tokenomics Specification

**Target path:** `docs/TOKENOMICS.md` in [`wrfcoin/wrfcomp`](https://github.com/wrfcoin)  
**Status:** Draft v0.1 (Shadow Leaderboard Specification)

> **Research draft, shadow only.** The four 2026-11-24 compute rules apply to the shadow simulation only. It does not
> replace WRFCoin's accepted tokenomics decisions or the skill pay that already exists on the testnet chain.
> The numbers are starting points for an off-chain paper ledger. Changing any slash or emission parameter
> here needs the project owner's approval.
>
> **This is not an offer of any token, reward or payment, and nothing pays out.** No one can buy, earn or
> receive anything from this document or from the shadow ledger.

---

## 1. Purpose & Rollout Strategy

WRFCoin redirects distributed compute and grassroots IoT hardware toward a single measurable public good: **lower-error, better-calibrated global and hyper-local weather forecasts**.

Rather than forcing chaotic atmospheric simulations to be bitwise-reproducible across heterogeneous GPUs for block consensus, WRFCoin decouples consensus from forecasting via **Proof-of-Forecast-Skill (PoFS)**. Participants are rewarded strictly for verifiable predictive skill and verified observational data quality, graded by the immutable `wrfcomp-eval` verification harness.

**Shadow-First Deployment Rule:** All reward, slashing, and emission mechanics defined in this specification MUST run as an **off-chain shadow ledger** (`src/wrfcomp/pofs_shadow.py`) against locked 2024–2025 historical hindcasts and live forward paper-epochs before any on-chain minting or slashing is activated.

---

## 2. Synoptic Epoch Lifecycle (Timelock Commit-Reveal)

Forecast epochs align with standard 6-hour global synoptic cycles (`00:00`, `06:00`, `12:00`, `18:00 UTC`). To eliminate forecast copying and selective non-reveal attacks, WRFCoin adopts a **timelock commit-reveal** cycle modeled on [Bittensor's Commit Reveal](https://learnbittensor.org/concepts/scoring-rewards/commit-reveal):

| Phase | Window | On-Chain / Storage Action | Security Property |
| :--- | :--- | :--- | :--- |
| **1. Commit** | `T0 - 15m` to `T0` | Compute nodes upload Zarr/Parquet ensemble payloads encrypted under a `drand` timelock key for `T0`, then post `(commit_sha256, content_cid, stake_amount)` on-chain. | Hides forecasts during submission so copycats cannot mirror top miners. |
| **2. Live Reveal** | `T0` | The `drand` beacon releases the round randomness; payloads decrypt automatically and merge into the live **WRFCoin Meta-Ensemble**. | Prevents selective non-reveal of bad forecasts and makes the Meta-Ensemble immediately usable before the weather occurs. |
| **3. Truth Window** | `T0 + 6h` to `T0 + 168h` | Ground-truth observations arrive from signed reference networks ([NOAA ISD/ASOS](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)) and high-trust WRFCoin IoT stations. | Multi-horizon scoring rewards both short-range nowcasting and medium-range skill. |
| **4. Settlement** | `T0 + 168h + 2h` | Validators run deterministic `wrfcomp-eval`, reach consensus on the Merkle root of scorecards, and trigger epoch minting and slashing. | Bitwise agreement is required only on the lightweight scoring CLI, never on GPU model training or inference. |

---

## 3. Epoch Emission Allocation

Each epoch mints `E_epoch` WRF (plus any protocol fee revenue from enterprise forecast API queries, split proportionally). Default launch parameters use the **Strict Anti-Copycat** configuration:

| Pool | Default Share | Target Participants | Primary Optimization Objective |
| :--- | :---: | :--- | :--- |
| **Compute Pool** (`P_comp`) | **55%** | WRFComp forecasting & downscaling nodes | Lower Meta-Ensemble CRPS & extreme-tail `twCRPS` |
| **Data Provider Pool** (`P_data`) | **35%** | WRFCoin & verified citizen IoT weather stations | High-QoD, spatially diverse, unspoofed surface observations |
| **Validator & Treasury Pool** (`P_val`) | **10%** | `wrfcomp-eval` verifier nodes & open archive ops | Deterministic truth ingestion & honest Merkle settlement |

---

## 4. Compute Pool Math (Skill + Marginal Ensemble Contribution)

Every participating compute node `i` posts a refundable stake `K_i >= K_min` for the epoch. Scoring uses strictly proper probabilistic rules implemented via the Bureau of Meteorology's [`scores`](https://github.com/nci/scores) library so miners maximize expected payout only by reporting their true calibrated probability distribution.

Admission follows the owner ruling of 2026-11-24. A timely recorder commitment and matching
reveal, stake at or above the floor, and passing gates against the free NOAA GEFS forecast
are required before a forecast enters the payout or marginal contribution pool. The
spread-skill band applies to every variable. Copies of GEFS earn zero. Among clones only the
first valid commit can earn. A committed non-reveal is slashed at the maximum honest rate.
The shadow defaults and replay procedure are in [SHADOW_LEDGER.md](SHADOW_LEDGER.md).

### 4.1 Individual Proper Skill (`CRPSS_i`)

For each evaluated variable `v` and lead time `h`, `wrfcomp-eval` computes the unbiased ("fair") ensemble CRPS and 95th-percentile threshold-weighted tail CRPS (`twCRPS_p95`), blended into a weighted proper loss:

```text
L_i = 0.70 * CRPS_fair(i) + 0.30 * twCRPS_p95(i)
```

Normalize against the free NOAA GEFS forecast for the same epoch and coverage:

```text
CRPSS_i = 1 - (L_i / L_base)
```

A miner that merely matches the free baseline scores `CRPSS_i = 0`. A miner worse than the baseline scores `CRPSS_i < 0`.

### 4.2 Marginal Ensemble Contribution (`Delta_CRPS_i`)

To prevent thousands of nodes from farming rewards by relaying identical copies of free public models, WRFCoin adapts [Numerai's Meta Model Contribution (MMC)](https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc) to probabilistic weather ensembles.

Let `M_all` be the equal-weight mixture across distinct eligible forecasts, and let `M_{-i}` be the Meta-Ensemble with miner `i` removed. Miner `i`'s marginal contribution is:

```text
Delta_CRPS_i = (L(M_{-i}) - L(M_all)) / L_base
```

- **Unique signal:** If removing miner `i` makes the Meta-Ensemble worse (`L(M_{-i}) > L(M_all)`), then `Delta_CRPS_i > 0`.
- **Clone / copycat:** GEFS relays and later clones are excluded before forming the mixture.

### 4.3 Composite Score, Payout & Slashing Cap

Miner `i`'s composite skill score blends individual skill and uniqueness using weight `w_delta` (default **`w_delta = 0.90`**):

```text
S_i = (1 - w_delta) * CRPSS_i + w_delta * Delta_CRPS_i
```

- **Positive Skill (`S_i > 0`):** Miner `i` receives a share of the Compute Pool proportional to its stake times its positive score:

```text
Reward_i = P_comp * (K_i * S_i) / sum_{j: S_j > 0} (K_j * S_j)
```

- **Negative Skill (`S_i < 0`):** Miner `i` earns `0` emission and burns a capped fraction of its stake (default **`max_slash_cap = 0.10`**, i.e., 10% max burn per epoch):

```text
Slash_i = min(max_slash_cap, alpha_slash * |S_i|) * K_i
```

- **Protocol Faults (Invalid Schema / Corrupt Ciphertext / Missed Reveal):** Maximum slash of `max_slash_cap * K_i` and exclusion from the Meta-Ensemble.

---

## 5. Data Provider Pool Math (Station QoD, PoL & Cell Capacity)

Weather station rewards adapt [WeatherXM's reward mechanism](https://docs.weatherxm.com/rewards/reward-mechanism) and wire directly into `wrfcomp/assimilation/qc.py` so the exact same trust math governs both AI assimilation weighting and token payouts.

For each registered station `s`, its epoch weight `W_s` is:

```text
W_s = QoD_s * PoL_s * CellCap_s * HW_s * Impact_s
```

Where:
1. **Quality of Data (`QoD_s in [0, 1]`):** Rolling reliability score emitted by `wrfcomp/assimilation/qc.py`, combining uptime, 4-gate MADIS/CrowdQC+ pass rate, bias-corrected innovation RMSE (`R_s`), daytime solar-heating penalty (`S_s`), and correlation/anti-spoof guard (`C_s`). Stations failing indoor (`r < 0.5`) or airport-copycat (`r > 0.998`, zero micro-residual) checks drop to `QoD_s = 0`.
2. **Proof of Location (`PoL_s in [0, 1]`):** Verifies hardware attestation, GPS consistency, and barometric pressure vs. elevation match. Frequent unexplained relocations decay `PoL_s`.
3. **Geospatial Cell Capacity (`CellCap_s`):** Using Uber H3 resolution-7 hex cells (~5.16 km^2), each cell has a maximum rewarded capacity `C_max` (default `1` for rural cells, up to `2` in complex urban/mountain terrain). If `N_cell > C_max` stations occupy the same cell, rewards dilute by `C_max / N_cell`.
4. **Hardware Class (`HW_s`):** Tier multiplier (e.g., `1.0` for standard temperature/pressure/humidity/wind stations; `1.25` for calibrated aspirated-shield + precipitation + solar radiation stations).
5. **Assimilation Impact Bonus (`Impact_s in [1.0, 1.5]`):** Periodic regional multiplier awarded to cells where assimilating local observations measurably lowers held-out regional CRPS in `wrfcomp-eval`.

Station `s` earns:

```text
Reward_s = P_data * (W_s / sum_{k} W_k)
```

---

## 6. Validator & Truth Oracle Economics

Validator nodes stake `K_val` and execute the deterministic, containerized `wrfcomp-eval` CLI over the epoch's revealed forecasts and signed truth snapshots:

- **Consensus:** Validators submit the SHA-256 Merkle root of the resulting `scorecards.json` array. Once a 2/3 stake supermajority agrees on the identical root, settlement finalizes and `P_val` splits proportionally among agreeing validators.
- **Equivocation / Divergence Slash:** A validator signing an invalid or non-canonical scorecard root loses `20%` of its validator stake.
- **Truth Integrity:** Ground truth is anchored first to signed public meteorological archives ([NOAA ISD/ASOS](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) and [ARCO-ERA5](https://github.com/google-research/arco-era5)); WRFCoin IoT stations are admitted into the truth set only after maintaining `QoD_s >= 0.90` and `PoL_s >= 0.95` for at least 30 consecutive days.

---

## 7. Threat Model & Anti-Gaming Controls

| Attack Vector | Mechanism Exploited | WRFCoin Mitigation |
| :--- | :--- | :--- |
| **Live Forecast Copycatting** | Copying top miner outputs before cutoff | `drand` timelock encryption until `T0` |
| **Free Public Model Relaying** | Submitting raw NOAA GFS or ECMWF AIFS clones | GEFS skill and calibration gates before marginal scoring |
| **Selective Non-Reveal** | Hiding bad forecasts after weather starts | Maximum slash on a committed non-reveal or invalid payload |
| **Sybil Stake Splitting** | Splitting one model across 50 wallets | Earliest-commit clone deduplication, minimum wallet stake and linear payout weight |
| **Backyard Station Farming** | Deploying 10 sensors at one address | H3 Cell Capacity (`C_max / N_cell`) + `PoL_s` |
| **Airport METAR Scraping** | Spoofing IoT readings from nearby ASOS feeds | Rolling `r > 0.998` + zero micro-residual anti-spoof guard (`C_s -> 0`) |
| **Calibration / Extreme-Event Shirking** | Hedging toward safe averages on calm days | Strictly proper CRPS + 30% weight on extreme-tail `twCRPS_p95` |

---

## 8. Tunable Governance Parameters

| Parameter | Default (Strict Anti-Copycat) | Safe Governance Range | Description |
| :--- | :---: | :---: | :--- |
| `epoch_length` | `6 hours` | `6h – 24h` | Synoptic forecast & commit-reveal cadence |
| `P_comp` | `55%` | `35% – 65%` | Share of epoch emissions for compute miners |
| `P_data` | `35%` | `25% – 55%` | Share of epoch emissions for weather stations |
| `P_val` | `10%` | `5% – 15%` | Share of epoch emissions for validators & treasury |
| `w_delta` | `0.90` | `0.65 – 0.95` | Weight on `Delta_CRPS` uniqueness vs. raw `CRPSS` |
| `twcrps_weight` | `0.30` | `0.20 – 0.50` | Weight on 95th-percentile extreme-tail CRPS |
| `max_slash_cap` | `10%` | `2% – 15%` | Maximum stake burn per epoch for negative skill |
| `h3_cell_cap` | `1` (rural) / `2` (complex) | `1 – 3` | Max un-diluted stations per H3 resolution-7 cell |
