"""Shared fixtures. Every test here is hermetic -- no ENTSO-E calls."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import PRICE_COL, fetch_day_ahead_prices, load_data_config
from src.model import BatteryConfig, OptimisationConfig

TZ = "Europe/Berlin"


class FakeEntsoeClient:
    """Stands in for EntsoePandasClient.

    Mimics the two behaviours of the real client that the code has to cope
    with: results come back in the zone's local timezone, and the end of the
    range is truncated *inclusively*.
    """

    def __init__(self, prices=None, freq="h", drop_at=None, tz=TZ):
        self.prices = prices
        self.freq = freq
        self.drop_at = pd.Timestamp(drop_at) if drop_at is not None else None
        self.tz = tz
        self.calls = 0

    def query_day_ahead_prices(self, zone, start, end):
        self.calls += 1
        index = pd.date_range(start, end, freq=self.freq, tz="UTC").tz_convert(self.tz)
        if self.prices is None:
            values = [50.0 + (i % 24) for i in range(len(index))]
        else:
            values = [self.prices[i % len(self.prices)] for i in range(len(index))]
        series = pd.Series(values, index=index, dtype="float64")
        if self.drop_at is not None and self.drop_at in series.index:
            series = series.drop(self.drop_at)
        return series


@pytest.fixture
def data_config():
    return load_data_config()


@pytest.fixture
def battery():
    """A 10 MW / 20 MWh asset with the round numbers the maths is checked against."""
    return BatteryConfig(
        power_charge_max_mw=10.0,
        power_discharge_max_mw=10.0,
        energy_capacity_mwh=20.0,
        soc_min_mwh=0.0,
        soc_max_mwh=20.0,
        soc_initial_mwh=0.0,
        efficiency_charge=0.92,
        efficiency_discharge=0.92,
        degradation_eur_per_mwh_discharged=3.0,
    )


@pytest.fixture
def optimisation():
    return OptimisationConfig(solver="appsi_highs", solver_tee=False)


@pytest.fixture
def price_frame():
    """Build a price frame directly, for model tests that need no DST realism."""

    def _make(prices, freq="h", start="2024-06-10 00:00+00:00"):
        index = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
        index.name = "timestamp_utc"
        hours = {"h": 1.0, "15min": 0.25, "30min": 0.5}[freq]
        local = index.tz_convert(TZ)
        return pd.DataFrame(
            {
                PRICE_COL: [float(p) for p in prices],
                "local_time": local,
                "delivery_date_local": local.normalize().tz_localize(None),
                "interval_hours": hours,
            },
            index=index,
        )

    return _make


@pytest.fixture
def fetched_frame():
    """Run the real ingestion path against a fake client, so DST logic is live."""

    def _fetch(start, end, **kwargs):
        client = kwargs.pop("client", None) or FakeEntsoeClient(**kwargs)
        return fetch_day_ahead_prices(start, end, client=client, use_cache=False)

    return _fetch
