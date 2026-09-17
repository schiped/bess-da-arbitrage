"""Gate-closure block structure, SoC continuity and the benchmark comparison."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.model import derive_terminal_value, solve_window
from src.rolling import (
    RollingConfig,
    build_blocks,
    compare_to_perfect_foresight,
    gate_closure_utc,
    horizon_slice,
    run_rolling,
)
from tests.conftest import TZ

# A repeating daily shape with a real spread, so every gate has a trade to make.
DAILY_SHAPE = [
    30, 25, 20, 18, 20, 35, 70, 110, 95, 60, 40, 30,
    25, 28, 35, 55, 90, 140, 160, 130, 95, 70, 50, 40,
]


@pytest.fixture
def rolling():
    return RollingConfig(
        gate_closure_local_time="12:00",
        commitment_days=1,
        lookahead_hours=24,
        lookahead_price_source="realised",
    )


@pytest.fixture
def week(fetched_frame):
    """A week straddling the October clock change, plus a lookahead tail."""
    return fetched_frame("2024-10-25", "2024-11-02", prices=DAILY_SHAPE)


# --------------------------------------------------------------------------
# Gate closure instants
# --------------------------------------------------------------------------

def test_gate_is_noon_local_on_the_day_before(rolling):
    gate = gate_closure_utc(pd.Timestamp("2024-06-10"), TZ, rolling)
    assert gate == pd.Timestamp("2024-06-09 10:00", tz="UTC")   # CEST, UTC+2


def test_gate_follows_the_clock_across_dst(rolling):
    """Noon local is 10:00Z under summer time and 11:00Z under winter time."""
    summer = gate_closure_utc(pd.Timestamp("2024-10-27"), TZ, rolling)
    winter = gate_closure_utc(pd.Timestamp("2024-10-28"), TZ, rolling)
    assert summer == pd.Timestamp("2024-10-26 10:00", tz="UTC")
    assert winter == pd.Timestamp("2024-10-27 11:00", tz="UTC")


def test_gate_always_precedes_delivery(week, rolling):
    blocks = build_blocks(week, TZ, rolling)
    for block in blocks:
        first_interval = week.index[block.commit_start]
        assert block.gate_closure_utc < first_interval


# --------------------------------------------------------------------------
# Block partitioning
# --------------------------------------------------------------------------

def test_blocks_follow_the_clock_not_a_fixed_count(week, rolling):
    blocks = build_blocks(week, TZ, rolling)
    committed = {
        block.delivery_days[0].date().isoformat(): block.n_committed
        for block in blocks
    }
    assert committed["2024-10-27"] == 25   # clocks go back
    assert committed["2024-10-26"] == 24
    assert committed["2024-10-28"] == 24


def test_march_block_is_23_intervals(fetched_frame, rolling):
    frame = fetched_frame("2024-03-30", "2024-04-01", prices=DAILY_SHAPE)
    blocks = build_blocks(frame, TZ, rolling)
    lengths = [b.n_committed for b in blocks]
    assert lengths == [24, 23, 24]


def test_blocks_tile_the_horizon_exactly_once(week, rolling):
    blocks = build_blocks(week, TZ, rolling)
    covered = []
    for block in blocks:
        covered.extend(range(block.commit_start, block.commit_stop))
    assert covered == list(range(len(week)))


def test_lookahead_extends_past_the_committed_block(week, rolling):
    blocks = build_blocks(week, TZ, rolling)
    # Every block but the last has a full day of tail available.
    assert blocks[0].n_lookahead == 24
    assert blocks[0].window_stop > blocks[0].commit_stop
    # The final block runs out of data and simply has no tail.
    assert blocks[-1].n_lookahead == 0


def test_zero_lookahead_means_window_equals_block(week, rolling):
    no_tail = dataclasses.replace(rolling, lookahead_hours=0)
    for block in build_blocks(week, TZ, no_tail):
        assert block.window_stop == block.commit_stop


def test_multi_day_commitment_blocks(week, rolling):
    two_day = dataclasses.replace(rolling, commitment_days=2)
    blocks = build_blocks(week, TZ, two_day)
    assert blocks[0].n_committed == 48
    assert len(blocks[0].delivery_days) == 2
    # The gate is still set by the first delivery day of the block.
    assert blocks[0].gate_closure_utc < week.index[0]


def test_horizon_days_limit_what_is_committed(week, rolling):
    wanted = {pd.Timestamp("2024-10-25"), pd.Timestamp("2024-10-26")}
    blocks = build_blocks(week, TZ, rolling, horizon_days=wanted)
    assert len(blocks) == 2
    # But the tail may still reach into days outside the horizon.
    assert blocks[-1].window_stop > blocks[-1].commit_stop


# --------------------------------------------------------------------------
# Running the horizon
# --------------------------------------------------------------------------

def test_state_of_charge_chains_across_gates(week, battery, optimisation, rolling):
    result = run_rolling(week, battery, optimisation, rolling, progress_every=0)
    blocks = result.blocks
    opens = blocks["soc_open_mwh"].to_numpy()
    closes = blocks["soc_close_mwh"].to_numpy()
    assert np.abs(opens[1:] - closes[:-1]).max() == pytest.approx(0.0, abs=1e-9)
    assert opens[0] == pytest.approx(battery.soc_initial_mwh)


def test_committed_schedule_covers_every_interval(week, battery, optimisation, rolling):
    result = run_rolling(week, battery, optimisation, rolling, progress_every=0)
    assert result.schedule.index.equals(week.index)
    assert result.summary["n_intervals"] == len(week)


def test_dst_day_is_committed_as_25_intervals(week, battery, optimisation, rolling):
    result = run_rolling(week, battery, optimisation, rolling, progress_every=0)
    row = result.blocks[
        result.blocks["delivery_date_local"] == pd.Timestamp("2024-10-27")
    ].iloc[0]
    assert row["n_committed"] == 25
    assert row["committed_hours"] == 25.0


def test_lookahead_tail_is_discarded_not_committed(
    week, battery, optimisation, rolling
):
    """Committed intervals must equal the horizon, not horizon plus tails."""
    result = run_rolling(week, battery, optimisation, rolling, progress_every=0)
    assert len(result.schedule) == len(week)
    assert result.blocks["n_committed"].sum() == len(week)


def test_flat_lookahead_changes_the_schedule(week, battery, optimisation, rolling):
    realised = run_rolling(week, battery, optimisation, rolling, progress_every=0)
    flat = run_rolling(
        week, battery, optimisation,
        dataclasses.replace(rolling, lookahead_price_source="flat"),
        progress_every=0,
    )
    assert not np.allclose(
        realised.schedule["soc_end_mwh"], flat.schedule["soc_end_mwh"]
    )


# --------------------------------------------------------------------------
# Benchmark comparison
# --------------------------------------------------------------------------

def test_rolling_never_beats_perfect_foresight(week, battery, optimisation, rolling):
    """The invariant. Perfect foresight optimises over a superset of the
    information and its schedule is feasible under any block structure, so no
    rolling variant can exceed it once both closing inventories are valued at
    the same rate."""
    horizon = horizon_slice(week, "2024-10-25", "2024-11-01")
    horizon_days = set(pd.unique(horizon["delivery_date_local"]))
    rate = derive_terminal_value(horizon, battery)
    perfect = solve_window(horizon, battery, optimisation)
    perfect_total = (
        perfect.summary["net_profit_eur"] + rate * perfect.summary["soc_final_mwh"]
    )

    for lookahead in (0, 6, 12, 24, 36):
        for source in ("realised", "flat"):
            config = dataclasses.replace(
                rolling, lookahead_hours=lookahead, lookahead_price_source=source
            )
            result = run_rolling(
                week, battery, optimisation, config,
                horizon_days=horizon_days, progress_every=0,
            )
            total = (
                result.summary["net_profit_eur"]
                + rate * result.summary["soc_final_mwh"]
            )
            assert total <= perfect_total + 1e-6, (
                f"lookahead={lookahead} source={source} beat perfect foresight"
            )


def test_more_lookahead_captures_more(week, battery, optimisation, rolling):
    horizon = horizon_slice(week, "2024-10-25", "2024-11-01")
    horizon_days = set(pd.unique(horizon["delivery_date_local"]))
    rate = derive_terminal_value(horizon, battery)

    totals = []
    for lookahead in (0, 12, 24):
        config = dataclasses.replace(rolling, lookahead_hours=lookahead)
        result = run_rolling(
            week, battery, optimisation, config,
            horizon_days=horizon_days, progress_every=0,
        )
        totals.append(
            result.summary["net_profit_eur"] + rate * result.summary["soc_final_mwh"]
        )
    assert totals[0] < totals[-1]
    assert totals == sorted(totals)


def test_comparison_refuses_an_impossible_result(week, battery, optimisation, rolling):
    horizon = horizon_slice(week, "2024-10-25", "2024-11-01")
    horizon_days = set(pd.unique(horizon["delivery_date_local"]))
    result = run_rolling(
        week, battery, optimisation, rolling,
        horizon_days=horizon_days, progress_every=0,
    )
    comparison = compare_to_perfect_foresight(result, horizon, battery, optimisation)
    assert 0.0 < comparison["capture_rate"] <= 1.0 + 1e-9
    assert comparison["gate_closure_cost_eur"] >= -1e-6


def test_per_gate_pnl_accounts_for_inventory(week, battery, optimisation, rolling):
    """A gate that ends holding stock is not loss-making just because the
    purchase landed in its window and the sale did not."""
    result = run_rolling(week, battery, optimisation, rolling, progress_every=0)
    blocks = result.blocks
    rebuilt = blocks["net_eur"] + blocks["inventory_value_change_eur"]
    pd.testing.assert_series_equal(
        rebuilt, blocks["net_incl_inventory_eur"], check_names=False
    )


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"commitment_days": 0}, "commitment_days"),
        ({"lookahead_hours": -1}, "lookahead_hours"),
        ({"lookahead_price_source": "vibes"}, "lookahead_price_source"),
        ({"gate_closure_local_time": "noon"}, "gate_closure_local_time"),
    ],
)
def test_invalid_rolling_config_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RollingConfig(**kwargs)
