"""LP economics, checked against hand-computed optima."""

from __future__ import annotations

import dataclasses

import pytest

from src.model import (
    InfeasibleWindowError,
    derive_terminal_value,
    exclusivity_intervals,
    solve_window,
)

ETA_C = ETA_D = 0.92
ETA_RT = ETA_C * ETA_D          # 0.8464
DEG = 3.0


# --------------------------------------------------------------------------
# Analytic optima
# --------------------------------------------------------------------------

def test_two_cheap_hours_then_two_dear_ones(price_frame, battery, optimisation):
    """Cheap-to-dear arbitrage, every number checkable by hand.

    Charging 10 MW for 2 h draws 20 MWh and stores 0.92 * 20 = 18.4 MWh, which
    is under the 20 MWh ceiling. Discharging that yields 18.4 * 0.92 = 16.928
    MWh at the meter, inside the 2 h * 10 MW delivery limit.
    """
    result = solve_window(
        price_frame([0, 0, 100, 100]), battery, optimisation,
        soc_initial_mwh=0.0, terminal_value_eur_per_mwh=0.0,
    )
    s = result.schedule
    assert s["energy_charged_mwh"].sum() == pytest.approx(20.0)
    assert s["soc_end_mwh"].max() == pytest.approx(18.4)
    assert s["energy_discharged_mwh"].sum() == pytest.approx(16.928)
    expected_net = 16.928 * 100 - 16.928 * DEG
    assert result.summary["net_profit_eur"] == pytest.approx(expected_net)


def test_spread_inside_the_round_trip_loss_produces_no_trade(
    price_frame, battery, optimisation
):
    result = solve_window(
        price_frame([100, 100, 110, 110]), battery, optimisation,
        soc_initial_mwh=0.0, terminal_value_eur_per_mwh=0.0,
    )
    assert result.schedule["energy_charged_mwh"].sum() == pytest.approx(0.0, abs=1e-6)
    assert result.summary["net_profit_eur"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("offset, expect_trade", [(-0.5, False), (+0.5, True)])
def test_break_even_uses_separate_efficiencies(
    price_frame, battery, optimisation, offset, expect_trade
):
    """Break-even must reflect eta_c and eta_d applied separately.

    Selling 1 MWh drawn returns ETA_RT MWh at the meter, less degradation on
    what is delivered, so the sell price has to clear
    (buy + DEG * ETA_D) / ETA_RT -- not buy / 0.92.
    """
    buy = 100.0
    break_even = (buy + DEG * ETA_D) / ETA_RT
    result = solve_window(
        price_frame([buy, buy, break_even + offset, break_even + offset]),
        battery, optimisation, soc_initial_mwh=0.0, terminal_value_eur_per_mwh=0.0,
    )
    traded = bool(result.schedule["energy_charged_mwh"].sum() > 1e-6)
    assert traded == expect_trade


def test_energy_scales_with_interval_length(price_frame, battery, optimisation):
    """Quarter-hourly MTUs must be MW * 0.25, not MW * 1."""
    result = solve_window(
        price_frame([0] * 8 + [100] * 4, freq="15min"), battery, optimisation,
        soc_initial_mwh=0.0, terminal_value_eur_per_mwh=0.0,
    )
    s = result.schedule
    # 4 quarters of discharge at 10 MW is 1 h, so 10 MWh is the binding limit.
    assert s["energy_discharged_mwh"].sum() == pytest.approx(10.0)
    # Free energy it cannot deliver is worthless, so it draws only what returns.
    assert s["energy_charged_mwh"].sum() == pytest.approx(10.0 / ETA_C / ETA_D)
    assert (s["energy_charged_mwh"] == s["charge_mw"] * 0.25).all()


# --------------------------------------------------------------------------
# Terminal value
# --------------------------------------------------------------------------

def test_terminal_value_derivation(price_frame, battery):
    prices = price_frame([10, 20, 30, 40])
    assert derive_terminal_value(prices, battery) == pytest.approx(
        ETA_D * (25.0 - DEG)
    )


def test_terminal_value_retains_energy_until_a_limit_binds(
    price_frame, battery, optimisation
):
    # 2 h at 10 MW stores 18.4 MWh, below the ceiling: energy-limited.
    short = solve_window(
        price_frame([10, 10]), battery, optimisation,
        soc_initial_mwh=0.0, terminal_value_eur_per_mwh=200.0,
    )
    assert short.soc_final_mwh == pytest.approx(18.4)

    # 4 h could store 36.8 MWh, so the SoC ceiling binds instead.
    long = solve_window(
        price_frame([10, 10, 10, 10]), battery, optimisation,
        soc_initial_mwh=0.0, terminal_value_eur_per_mwh=200.0,
    )
    assert long.soc_final_mwh == pytest.approx(battery.soc_max_mwh)


def test_zero_terminal_value_empties_the_battery(price_frame, battery, optimisation):
    """The artefact the terminal term exists to prevent."""
    result = solve_window(
        price_frame([10, 10, 10, 10]), battery, optimisation,
        soc_initial_mwh=20.0, terminal_value_eur_per_mwh=0.0,
    )
    assert result.soc_final_mwh == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Simultaneous charge and discharge at negative prices
# --------------------------------------------------------------------------

def test_in_place_cycling_threshold(battery):
    expected = -ETA_RT * DEG / (1 - ETA_RT)
    assert battery.in_place_cycling_price_threshold == pytest.approx(expected)
    assert expected == pytest.approx(-16.5312, abs=1e-3)


def test_lossless_battery_can_never_cycle_in_place(battery):
    lossless = dataclasses.replace(
        battery, efficiency_charge=1.0, efficiency_discharge=1.0
    )
    assert lossless.in_place_cycling_price_threshold == float("-inf")


def test_only_deeply_negative_intervals_get_binaries(price_frame, battery):
    prices = price_frame([-500, -10, 0, 50])
    assert exclusivity_intervals(prices, battery) == [0]


def test_full_battery_at_negative_prices_does_not_cycle_in_place(
    price_frame, battery, optimisation
):
    """A full battery paid to consume must not burn energy to make room.

    Charging at full power while discharging just enough to hold SoC flat is
    profitable below the threshold, but a single-inverter asset cannot do it.
    """
    prices = price_frame([-500, -500, 100, 100])
    result = solve_window(
        prices, battery, optimisation,
        soc_initial_mwh=battery.soc_max_mwh, terminal_value_eur_per_mwh=0.0,
    )
    overlap = result.schedule[["charge_mw", "discharge_mw"]].min(axis=1).max()
    assert overlap == pytest.approx(0.0, abs=1e-6)
    assert result.summary["exclusivity_binaries"] == 2


def test_guard_catches_in_place_cycling_when_enforcement_is_off(
    price_frame, battery, optimisation
):
    unenforced = dataclasses.replace(
        optimisation, enforce_no_simultaneous_operation=False
    )
    with pytest.raises(InfeasibleWindowError, match="overlap"):
        solve_window(
            price_frame([-500, -500, 100, 100]), battery, unenforced,
            soc_initial_mwh=battery.soc_max_mwh, terminal_value_eur_per_mwh=0.0,
        )


def test_windows_without_negative_prices_stay_a_pure_lp(
    price_frame, battery, optimisation
):
    result = solve_window(
        price_frame([20, 30, 90, 120]), battery, optimisation, soc_initial_mwh=0.0
    )
    assert result.summary["exclusivity_binaries"] == 0


# --------------------------------------------------------------------------
# Physical limits
# --------------------------------------------------------------------------

def test_schedule_respects_power_and_soc_limits(price_frame, battery, optimisation):
    prices = price_frame([5, 5, 200, 5, 5, 200, 5, 200])
    result = solve_window(prices, battery, optimisation, soc_initial_mwh=10.0)
    s = result.schedule
    assert s["charge_mw"].max() <= battery.power_charge_max_mw + 1e-9
    assert s["discharge_mw"].max() <= battery.power_discharge_max_mw + 1e-9
    assert s["soc_end_mwh"].max() <= battery.soc_max_mwh + 1e-9
    assert s["soc_end_mwh"].min() >= battery.soc_min_mwh - 1e-9


def test_state_of_charge_balance_closes(price_frame, battery, optimisation):
    prices = price_frame([5, 5, 200, 5, 5, 200, 5, 200])
    result = solve_window(prices, battery, optimisation, soc_initial_mwh=10.0)
    s = result.schedule
    stored = (
        battery.efficiency_charge * s["energy_charged_mwh"].sum()
        - s["energy_discharged_mwh"].sum() / battery.efficiency_discharge
    )
    assert 10.0 + stored == pytest.approx(s["soc_end_mwh"].iloc[-1], abs=1e-9)


def test_dst_days_solve_at_their_true_length(fetched_frame, battery, optimisation):
    for day, expected in [("2024-10-27", 25), ("2024-03-31", 23)]:
        prices = fetched_frame(day, day, prices=[20, 30, 90, 120, 40, 15])
        result = solve_window(prices, battery, optimisation, soc_initial_mwh=10.0)
        assert len(result.schedule) == expected
        assert result.summary["span_hours"] == float(expected)


def test_opening_soc_outside_bounds_is_rejected(price_frame, battery, optimisation):
    with pytest.raises(ValueError, match="outside"):
        solve_window(
            price_frame([10, 20]), battery, optimisation, soc_initial_mwh=999.0
        )


def test_missing_columns_are_reported(price_frame, battery, optimisation):
    prices = price_frame([10, 20]).drop(columns=["interval_hours"])
    with pytest.raises(ValueError, match="interval_hours"):
        solve_window(prices, battery, optimisation)
