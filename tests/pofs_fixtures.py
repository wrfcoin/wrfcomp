"""Synthetic recorder and single-field GEFS epoch for settlement tests only."""

import numpy as np

from wrfcomp.pofs_shadow import Commitment, EpochEvidence, forecast_digest, settle_compute_pool


def recorded_epoch(miners, **kwargs):
    # The test recorder accepts each forecast in list order before the first init.
    # Withholding cases retain their receipt but omit the revealed payload.
    t = np.datetime64("2024-01-01T00:00:00", "ns")
    kwargs.setdefault("tail_threshold", float(np.quantile(kwargs["obs"], 0.95)))
    receipts = {
        m.miner_id: Commitment(
            forecast_digest(m.forecast) if m.forecast is not None and not m.protocol_fault else "0" * 64,
            t - np.timedelta64(len(miners) - i, "s"),
        )
        for i, m in enumerate(miners)
    }
    anchor = kwargs.get("gefs_forecast", kwargs["baseline_forecast"])
    evidence = EpochEvidence(t, forecast_digest(anchor), receipts)
    return settle_compute_pool(
        miners,
        evidence=evidence,
        issue_times=t + np.arange(len(anchor)) * np.timedelta64(6, "h"),
        lead_hours=6,
        **kwargs,
    )
