"""Ingestion, DST handling and cache behaviour."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import (
    PRICE_COL,
    PriceDataError,
    _request_chunks,
    cache_path,
    delivery_window_utc,
    fetch_day_ahead_prices,
    load_data_config,
)
from tests.conftest import TZ, FakeEntsoeClient


# --------------------------------------------------------------------------
# Delivery windows
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "start, end, expected_hours",
    [
        ("2024-10-27", "2024-10-27", 25),   # clocks go back, 25-hour local day
        ("2024-03-31", "2024-03-31", 23),   # clocks go forward, 23-hour local day
        ("2023-10-29", "2023-10-29", 25),
        ("2023-03-26", "2023-03-26", 23),
        ("2024-06-10", "2024-06-10", 24),
        ("2023-01-01", "2023-12-31", 365 * 24),
        ("2023-01-01", "2024-12-31", (365 + 366) * 24),
    ],
)
def test_delivery_window_spans_absorb_clock_changes(start, end, expected_hours):
    utc_start, utc_end = delivery_window_utc(start, end, TZ)
    assert (utc_end - utc_start) == pd.Timedelta(hours=expected_hours)


def test_delivery_window_is_half_open_in_utc():
    _, first_end = delivery_window_utc("2024-06-10", "2024-06-10", TZ)
    second_start, _ = delivery_window_utc("2024-06-11", "2024-06-11", TZ)
    assert first_end == second_start


def test_reversed_range_is_rejected():
    with pytest.raises(ValueError, match="precedes"):
        delivery_window_utc("2024-06-10", "2024-06-01", TZ)


def test_timestamp_with_time_of_day_maps_to_its_delivery_day():
    with_time = delivery_window_utc("2024-06-10 17:30", "2024-06-10 03:00", TZ)
    plain = delivery_window_utc("2024-06-10", "2024-06-10", TZ)
    assert with_time == plain


# --------------------------------------------------------------------------
# DST as it lands in the returned frame
# --------------------------------------------------------------------------

def test_october_day_has_25_intervals_and_a_repeated_local_hour(fetched_frame):
    df = fetched_frame("2024-10-27", "2024-10-27")
    assert len(df) == 25
    assert df["interval_hours"].sum() == 25.0
    # UTC stays unique and ordered even though the wall clock repeats 02:00.
    assert df.index.is_unique and df.index.is_monotonic_increasing
    local_hours = df["local_time"].dt.strftime("%H:%M").tolist()
    assert local_hours.count("02:00") == 2
    assert df["delivery_date_local"].nunique() == 1


def test_march_day_has_23_intervals_and_no_local_02h(fetched_frame):
    df = fetched_frame("2024-03-31", "2024-03-31")
    assert len(df) == 23
    assert df["interval_hours"].sum() == 23.0
    assert (df["local_time"].dt.strftime("%H:%M") == "02:00").sum() == 0


def test_quarter_hourly_october_day_has_100_intervals(fetched_frame):
    df = fetched_frame("2024-10-27", "2024-10-27", freq="15min")
    assert len(df) == 100
    assert df["interval_hours"].sum() == 25.0
    assert (df["interval_hours"] == 0.25).all()


def test_year_of_days_is_not_24_times_n(fetched_frame):
    df = fetched_frame("2024-03-30", "2024-04-01")
    hours_per_day = df.groupby("delivery_date_local")["interval_hours"].sum()
    assert hours_per_day.tolist() == [24.0, 23.0, 24.0]
    assert len(df) == 71  # not 72


# --------------------------------------------------------------------------
# Request chunking
# --------------------------------------------------------------------------

def test_long_ranges_are_chunked(data_config):
    utc_start, utc_end = delivery_window_utc("2023-01-01", "2024-12-31", TZ)
    chunks = _request_chunks(utc_start, utc_end, data_config)
    assert len(chunks) > 1
    # Every chunk stays under the library's one-year internal split threshold.
    for chunk_start, chunk_stop in chunks:
        assert (chunk_stop - chunk_start) < pd.Timedelta(days=365)


def test_chunk_boundaries_do_not_open_holes(fetched_frame):
    """Regression: entsoe-py drops the interval on its own year-split boundary.

    A two-year pull must still tile the window exactly, with no missing hour
    at any chunk seam.
    """
    client = FakeEntsoeClient()
    df = fetched_frame("2023-01-01", "2024-12-31", client=client)
    assert client.calls > 1, "range should have been split into several requests"
    assert len(df) == 17544
    assert df["interval_hours"].sum() == 17544.0
    assert df.index.is_unique

    utc_start, utc_end = delivery_window_utc("2023-01-01", "2024-12-31", TZ)
    expected = pd.date_range(utc_start, utc_end, freq="h", inclusive="left")
    assert df.index.equals(expected)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_missing_interval_is_rejected(fetched_frame):
    with pytest.raises(PriceDataError, match="unexpected interval lengths"):
        fetched_frame("2024-06-10", "2024-06-10", drop_at="2024-06-10 05:00+00:00")


def test_missing_interval_only_warns_when_not_strict(fetched_frame, recwarn):
    config = load_data_config(strict=False)
    client = FakeEntsoeClient(drop_at="2024-06-10 05:00+00:00")
    df = fetch_day_ahead_prices(
        "2024-06-10", "2024-06-10", config=config, client=client, use_cache=False
    )
    assert len(df) == 23  # the hole is reported, not repaired


def test_short_window_needs_two_intervals_to_infer_resolution(fetched_frame):
    with pytest.raises(PriceDataError):
        fetched_frame("2024-06-10", "2024-06-10", freq="D")


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def test_cache_round_trips_and_skips_the_network(tmp_path):
    config = load_data_config(cache_dir=tmp_path)
    client = FakeEntsoeClient()

    first = fetch_day_ahead_prices("2024-10-27", "2024-10-27", config=config, client=client)
    calls_after_fetch = client.calls
    assert calls_after_fetch > 0

    path = cache_path("2024-10-27", "2024-10-27", config)
    assert path.exists()
    assert path.with_suffix(".meta.json").exists()

    class Exploding:
        def query_day_ahead_prices(self, *args, **kwargs):
            raise AssertionError("cache hit must not reach the network")

    second = fetch_day_ahead_prices(
        "2024-10-27", "2024-10-27", config=config, client=Exploding()
    )
    assert second.index.equals(first.index)
    assert second.index.tz is not None and str(second.index.tz) == "UTC"
    pd.testing.assert_series_equal(second[PRICE_COL], first[PRICE_COL])


def test_cache_key_separates_zones_and_ranges(tmp_path):
    config = load_data_config(cache_dir=tmp_path)
    other_zone = load_data_config(cache_dir=tmp_path, zone="NL")
    a = cache_path("2024-01-01", "2024-01-31", config)
    b = cache_path("2024-02-01", "2024-02-28", config)
    c = cache_path("2024-01-01", "2024-01-31", other_zone)
    assert len({a, b, c}) == 3


def test_force_refresh_requeries(tmp_path):
    config = load_data_config(cache_dir=tmp_path)
    client = FakeEntsoeClient()
    fetch_day_ahead_prices("2024-06-10", "2024-06-10", config=config, client=client)
    before = client.calls
    fetch_day_ahead_prices(
        "2024-06-10", "2024-06-10", config=config, client=client, force_refresh=True
    )
    assert client.calls > before
