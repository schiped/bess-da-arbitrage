"""Single-window day-ahead arbitrage LP for a battery.

Solves one optimisation window: given a price frame from :mod:`src.data`, an
asset definition and an opening state of charge, find the charge/discharge
schedule that maximises net revenue. It knows nothing about delivery days, gate
closure or rolling horizons -- that scheduling logic belongs to the caller, so
this model can be pointed at a day, a week or two whole years unchanged.

Formulation
-----------
Decision variables per interval ``t`` of duration ``h_t`` hours:

* ``charge_mw[t]``    -- grid draw, MW, in ``[0, power_charge_max_mw]``
* ``discharge_mw[t]`` -- grid injection, MW, in ``[0, power_discharge_max_mw]``
* ``soc_mwh[t]``      -- state of charge at the **end** of interval t, MWh

State of charge balance, with the two efficiencies applied separately rather
than lumped into a round-trip figure::

    soc[t] = soc[t-1] + eta_charge * charge_mw[t] * h_t
                      - discharge_mw[t] * h_t / eta_discharge

Objective, maximised::

    sum_t ( price[t] * discharge_mw[t] * h_t          <- revenue
          - price[t] * charge_mw[t] * h_t             <- purchase cost
          - degradation * discharge_mw[t] * h_t )     <- throughput cost
    + terminal_value * soc[last]

Every energy term is ``MW * h_t``, never ``MW * 1``, so 23- and 25-hour
delivery days and the 2025 switch to quarter-hourly MTUs need no special case.

Simultaneous charge and discharge
--------------------------------
At non-negative prices the LP relaxation is naturally exclusive: paying to
inject while being paid to withdraw at the same price cannot beat the
round-trip loss. Deeply negative prices break that. A full battery facing
-500 EUR/MWh can charge at full power while discharging just enough to hold SoC
flat, burning energy through the round-trip loss purely to get paid more for
consuming. Per MWh drawn that nets ``-(1 - eta_rt) * price - eta_rt * deg``, so
it pays whenever::

    price < -eta_rt * degradation / (1 - eta_rt)      (-16.53 EUR/MWh by default)

DE-LU cleared below that repeatedly in 2023-24 (floor -500), so it is not a
corner case. A single-inverter asset cannot do it, so
:func:`build_window_model` adds an exclusivity binary -- but *only* on the
intervals priced below that threshold, where the relaxation is actually loose.
Everywhere else the LP is already exact, so a two-year window stays a handful
of binaries rather than tens of thousands, and most windows stay pure LPs.

Usage
-----
    py -m src.model --start 2024-10-25 --end 2024-10-28
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import pyomo.environ as pyo
import yaml

from src.data import INDEX_NAME, PRICE_COL, fetch_day_ahead_prices

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATTERY_CONFIG = PROJECT_ROOT / "config" / "battery.yaml"
DEFAULT_OPTIMISATION_CONFIG = PROJECT_ROOT / "config" / "optimisation.yaml"


class InfeasibleWindowError(RuntimeError):
    """Raised when the solver does not return an optimal schedule."""


@dataclass(frozen=True)
class BatteryConfig:
    """Asset parameters. Power in MW, energy and SoC in MWh."""

    power_charge_max_mw: float
    power_discharge_max_mw: float
    energy_capacity_mwh: float
    soc_min_mwh: float
    soc_max_mwh: float
    soc_initial_mwh: float
    efficiency_charge: float
    efficiency_discharge: float
    degradation_eur_per_mwh_discharged: float

    def __post_init__(self) -> None:
        if not 0.0 < self.efficiency_charge <= 1.0:
            raise ValueError("efficiency_charge must be in (0, 1]")
        if not 0.0 < self.efficiency_discharge <= 1.0:
            raise ValueError("efficiency_discharge must be in (0, 1]")
        if self.soc_min_mwh >= self.soc_max_mwh:
            raise ValueError("soc_min_mwh must be below soc_max_mwh")
        if self.soc_max_mwh > self.energy_capacity_mwh:
            raise ValueError("soc_max_mwh exceeds energy_capacity_mwh")
        if not self.soc_min_mwh <= self.soc_initial_mwh <= self.soc_max_mwh:
            raise ValueError("soc_initial_mwh lies outside the usable SoC window")

    @property
    def round_trip_efficiency(self) -> float:
        return self.efficiency_charge * self.efficiency_discharge

    @property
    def in_place_cycling_price_threshold(self) -> float:
        """Price below which charging and discharging at once turns a profit.

        Holding SoC flat while drawing 1 MWh from the grid delivers
        ``eta_rt`` MWh back, so the interval nets
        ``-(1 - eta_rt) * price - eta_rt * degradation``. That is positive
        below this price. A lossless battery can never do it, hence -inf.
        """
        eta_rt = self.round_trip_efficiency
        if eta_rt >= 1.0:
            return float("-inf")
        return -eta_rt * self.degradation_eur_per_mwh_discharged / (1.0 - eta_rt)


@dataclass(frozen=True)
class OptimisationConfig:
    """Solver and objective settings."""

    solver: str = "appsi_highs"
    solver_tee: bool = False
    terminal_soc_value_eur_per_mwh: Optional[float] = None
    results_dir: str = "results"
    max_simultaneous_mw: float = 1e-6
    enforce_no_simultaneous_operation: bool = True

    @property
    def results_root(self) -> Path:
        path = Path(self.results_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_battery_config(path: Optional[Union[str, Path]] = None, **overrides) -> BatteryConfig:
    """Load the asset definition from YAML."""
    config_path = Path(path) if path is not None else DEFAULT_BATTERY_CONFIG
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw.update(overrides)
    return BatteryConfig(**raw)


def load_optimisation_config(
    path: Optional[Union[str, Path]] = None, **overrides
) -> OptimisationConfig:
    """Load solver/objective settings from YAML."""
    config_path = Path(path) if path is not None else DEFAULT_OPTIMISATION_CONFIG
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw.update(overrides)
    return OptimisationConfig(**raw)


@dataclass
class WindowResult:
    """Outcome of a single solved window."""

    schedule: pd.DataFrame
    summary: dict
    terminal_soc_value_eur_per_mwh: float
    solver_status: str
    solver_seconds: float

    @property
    def soc_final_mwh(self) -> float:
        """Closing SoC -- the opening SoC of the next window in a rolling run."""
        return float(self.schedule["soc_end_mwh"].iloc[-1])


# ---------------------------------------------------------------------------
# Objective terms
# ---------------------------------------------------------------------------

def derive_terminal_value(prices: pd.DataFrame, battery: BatteryConfig) -> float:
    """Default value of a stored MWh at the horizon edge, in EUR/MWh stored.

    One MWh sitting in the battery yields ``efficiency_discharge`` MWh at the
    meter, each earning the prevailing price less the throughput cost. Using
    the window's own mean price keeps the term self-scaling across seasons
    instead of pinning a number that goes stale.
    """
    mean_price = float(prices[PRICE_COL].mean())
    return battery.efficiency_discharge * (
        mean_price - battery.degradation_eur_per_mwh_discharged
    )


def exclusivity_intervals(prices: pd.DataFrame, battery: BatteryConfig) -> list:
    """Positions where simultaneous charge/discharge could be profitable.

    Only these need an exclusivity binary; elsewhere the LP relaxation already
    yields mutually exclusive operation, so there is nothing to enforce.
    """
    threshold = battery.in_place_cycling_price_threshold
    price = prices[PRICE_COL].to_numpy(dtype="float64")
    # Small margin so an interval sitting exactly on the threshold is covered.
    return [t for t in range(len(price)) if price[t] <= threshold + 1e-9]


def build_window_model(
    prices: pd.DataFrame,
    battery: BatteryConfig,
    soc_initial_mwh: float,
    terminal_value_eur_per_mwh: float,
    enforce_exclusivity: bool = True,
) -> pyo.ConcreteModel:
    """Construct the Pyomo model for one window."""
    if prices.empty:
        raise ValueError("cannot optimise an empty price window")

    price = prices[PRICE_COL].to_numpy(dtype="float64")
    hours = prices["interval_hours"].to_numpy(dtype="float64")
    n = len(prices)

    m = pyo.ConcreteModel(name="bess_day_ahead_arbitrage")
    m.T = pyo.RangeSet(0, n - 1)

    m.charge_mw = pyo.Var(m.T, bounds=(0.0, battery.power_charge_max_mw))
    m.discharge_mw = pyo.Var(m.T, bounds=(0.0, battery.power_discharge_max_mw))
    m.soc_mwh = pyo.Var(m.T, bounds=(battery.soc_min_mwh, battery.soc_max_mwh))

    # Exclusivity binaries, only where deeply negative prices make in-place
    # cycling attractive. The model stays a pure LP when that set is empty.
    exclusive = exclusivity_intervals(prices, battery) if enforce_exclusivity else []
    if exclusive:
        m.E = pyo.Set(initialize=exclusive, ordered=True)
        m.is_charging = pyo.Var(m.E, domain=pyo.Binary)

        def _charge_exclusive(model, t):
            return model.charge_mw[t] <= battery.power_charge_max_mw * model.is_charging[t]

        def _discharge_exclusive(model, t):
            return model.discharge_mw[t] <= battery.power_discharge_max_mw * (
                1 - model.is_charging[t]
            )

        m.charge_exclusive = pyo.Constraint(m.E, rule=_charge_exclusive)
        m.discharge_exclusive = pyo.Constraint(m.E, rule=_discharge_exclusive)

    def _soc_balance(model, t):
        opening = soc_initial_mwh if t == 0 else model.soc_mwh[t - 1]
        return model.soc_mwh[t] == (
            opening
            + battery.efficiency_charge * model.charge_mw[t] * hours[t]
            - model.discharge_mw[t] * hours[t] / battery.efficiency_discharge
        )

    m.soc_balance = pyo.Constraint(m.T, rule=_soc_balance)

    trading = sum(
        (price[t] - battery.degradation_eur_per_mwh_discharged)
        * m.discharge_mw[t]
        * hours[t]
        - price[t] * m.charge_mw[t] * hours[t]
        for t in range(n)
    )
    m.objective = pyo.Objective(
        expr=trading + terminal_value_eur_per_mwh * m.soc_mwh[n - 1],
        sense=pyo.maximize,
    )
    return m


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _extract_schedule(
    m: pyo.ConcreteModel, prices: pd.DataFrame, battery: BatteryConfig
) -> pd.DataFrame:
    """Pull the solution into a per-interval DataFrame."""
    n = len(prices)
    hours = prices["interval_hours"].to_numpy(dtype="float64")

    schedule = pd.DataFrame(index=prices.index.copy())
    schedule.index.name = INDEX_NAME
    for column in ("local_time", "delivery_date_local"):
        if column in prices.columns:
            schedule[column] = prices[column]
    schedule[PRICE_COL] = prices[PRICE_COL]
    schedule["interval_hours"] = hours

    # Simplex returns values like -1e-17 and signed zeros at the bounds. Snap
    # them so downstream sums, CSVs and SoC hand-offs read cleanly.
    def _clean(values):
        return [0.0 if abs(v) < 1e-9 else float(v) for v in values]

    schedule["charge_mw"] = _clean(pyo.value(m.charge_mw[t]) for t in range(n))
    schedule["discharge_mw"] = _clean(pyo.value(m.discharge_mw[t]) for t in range(n))
    schedule["soc_end_mwh"] = _clean(pyo.value(m.soc_mwh[t]) for t in range(n))

    schedule["energy_charged_mwh"] = schedule["charge_mw"] * schedule["interval_hours"]
    schedule["energy_discharged_mwh"] = schedule["discharge_mw"] * schedule["interval_hours"]

    schedule["charge_cost_eur"] = schedule[PRICE_COL] * schedule["energy_charged_mwh"]
    schedule["discharge_revenue_eur"] = schedule[PRICE_COL] * schedule["energy_discharged_mwh"]
    schedule["degradation_cost_eur"] = (
        battery.degradation_eur_per_mwh_discharged * schedule["energy_discharged_mwh"]
    )
    schedule["net_eur"] = (
        schedule["discharge_revenue_eur"]
        - schedule["charge_cost_eur"]
        - schedule["degradation_cost_eur"]
    )
    schedule["cumulative_net_eur"] = schedule["net_eur"].cumsum()
    return schedule


def _summarise(
    schedule: pd.DataFrame,
    battery: BatteryConfig,
    soc_initial_mwh: float,
    terminal_value: float,
) -> dict:
    """Headline economics for the window."""
    discharged = float(schedule["energy_discharged_mwh"].sum())
    charged = float(schedule["energy_charged_mwh"].sum())
    net = float(schedule["net_eur"].sum())
    soc_final = float(schedule["soc_end_mwh"].iloc[-1])
    span_hours = float(schedule["interval_hours"].sum())

    return {
        "n_intervals": int(len(schedule)),
        "span_hours": span_hours,
        "first_interval_utc": schedule.index[0].isoformat(),
        "last_interval_utc": schedule.index[-1].isoformat(),
        "mean_price_eur_mwh": float(schedule[PRICE_COL].mean()),
        "energy_charged_mwh": charged,
        "energy_discharged_mwh": discharged,
        "equivalent_full_cycles": discharged / battery.energy_capacity_mwh,
        "cycles_per_day": (discharged / battery.energy_capacity_mwh) / (span_hours / 24.0),
        "gross_revenue_eur": float(schedule["discharge_revenue_eur"].sum()),
        "charge_cost_eur": float(schedule["charge_cost_eur"].sum()),
        "degradation_cost_eur": float(schedule["degradation_cost_eur"].sum()),
        "net_profit_eur": net,
        "net_profit_eur_per_mw_year": (
            net / battery.power_discharge_max_mw / (span_hours / 8760.0)
        ),
        "soc_initial_mwh": float(soc_initial_mwh),
        "soc_final_mwh": soc_final,
        "terminal_soc_value_eur_per_mwh": float(terminal_value),
        "terminal_soc_value_eur": float(terminal_value * soc_final),
        "round_trip_efficiency": battery.round_trip_efficiency,
    }


def solve_window(
    prices: pd.DataFrame,
    battery: Optional[BatteryConfig] = None,
    optimisation: Optional[OptimisationConfig] = None,
    soc_initial_mwh: Optional[float] = None,
    terminal_value_eur_per_mwh: Optional[float] = None,
) -> WindowResult:
    """Optimise one price window and return the schedule plus economics.

    Parameters
    ----------
    prices
        Frame from :func:`src.data.fetch_day_ahead_prices`, or any slice of one.
        Must carry ``price_eur_mwh`` and ``interval_hours``.
    battery, optimisation
        Configs; default to ``config/battery.yaml`` and
        ``config/optimisation.yaml``.
    soc_initial_mwh
        Opening SoC. Defaults to the battery config's value; a rolling caller
        passes the previous window's closing SoC here.
    terminal_value_eur_per_mwh
        Value of stored energy at the horizon edge. Falls back to the
        optimisation config, then to :func:`derive_terminal_value`.
    """
    battery = battery or load_battery_config()
    optimisation = optimisation or load_optimisation_config()

    missing = {PRICE_COL, "interval_hours"} - set(prices.columns)
    if missing:
        raise ValueError(f"price frame is missing column(s): {sorted(missing)}")

    soc_open = battery.soc_initial_mwh if soc_initial_mwh is None else float(soc_initial_mwh)
    if not battery.soc_min_mwh - 1e-9 <= soc_open <= battery.soc_max_mwh + 1e-9:
        raise ValueError(
            f"opening SoC {soc_open:g} MWh lies outside "
            f"[{battery.soc_min_mwh:g}, {battery.soc_max_mwh:g}]"
        )

    terminal_value = terminal_value_eur_per_mwh
    if terminal_value is None:
        terminal_value = optimisation.terminal_soc_value_eur_per_mwh
    if terminal_value is None:
        terminal_value = derive_terminal_value(prices, battery)
    terminal_value = float(terminal_value)

    exclusive = (
        exclusivity_intervals(prices, battery)
        if optimisation.enforce_no_simultaneous_operation
        else []
    )
    if exclusive:
        LOGGER.info(
            "%d of %d intervals priced at or below %.2f EUR/MWh; adding exclusivity "
            "binaries so the battery cannot cycle in place",
            len(exclusive), len(prices), battery.in_place_cycling_price_threshold,
        )

    model = build_window_model(
        prices,
        battery,
        soc_open,
        terminal_value,
        enforce_exclusivity=optimisation.enforce_no_simultaneous_operation,
    )

    solver = pyo.SolverFactory(optimisation.solver)
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            f"solver '{optimisation.solver}' is not available. "
            "Install highspy (pip install highspy) or point config/optimisation.yaml "
            "at another solver."
        )

    started = datetime.now(timezone.utc)
    LOGGER.info(
        "solving %d intervals (%g h), opening SoC %.3f MWh, terminal value %.2f EUR/MWh",
        len(prices), float(prices["interval_hours"].sum()), soc_open, terminal_value,
    )
    outcome = solver.solve(model, tee=optimisation.solver_tee)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    condition = outcome.solver.termination_condition
    if condition != pyo.TerminationCondition.optimal:
        raise InfeasibleWindowError(f"solver terminated with status '{condition}'")

    schedule = _extract_schedule(model, prices, battery)

    # Belt and braces: the binaries above should make this impossible, and
    # where they are absent the relaxation should be exact. A non-zero overlap
    # would mean reported round-trip energy that no inverter ever moved.
    overlap = float(schedule[["charge_mw", "discharge_mw"]].min(axis=1).max())
    if overlap > optimisation.max_simultaneous_mw:
        raise InfeasibleWindowError(
            f"charge and discharge overlap by {overlap:g} MW in at least one interval. "
            f"Below {battery.in_place_cycling_price_threshold:.2f} EUR/MWh the battery "
            "profits from cycling in place, which a single-inverter asset cannot do; "
            "set enforce_no_simultaneous_operation: true in config/optimisation.yaml"
        )

    summary = _summarise(schedule, battery, soc_open, terminal_value)
    summary["objective_eur"] = float(pyo.value(model.objective))
    summary["solver"] = optimisation.solver
    summary["solver_seconds"] = elapsed
    summary["exclusivity_binaries"] = len(exclusive)
    summary["in_place_cycling_price_threshold_eur_mwh"] = (
        battery.in_place_cycling_price_threshold
    )
    summary["max_simultaneous_overlap_mw"] = overlap

    LOGGER.info(
        "solved in %.2f s: net %.2f EUR, %.2f EFC, closing SoC %.3f MWh",
        elapsed, summary["net_profit_eur"], summary["equivalent_full_cycles"],
        summary["soc_final_mwh"],
    )
    return WindowResult(
        schedule=schedule,
        summary=summary,
        terminal_soc_value_eur_per_mwh=terminal_value,
        solver_status=str(condition),
        solver_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Run outputs
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    """Current commit, so a results folder can be traced back to the code."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def write_run_outputs(
    result: WindowResult,
    battery: BatteryConfig,
    optimisation: OptimisationConfig,
    provenance: Optional[dict] = None,
    run_name: Optional[str] = None,
) -> Path:
    """Write the schedule CSV and the assumptions dump for one run.

    Returns the run directory. Every run gets its own timestamped folder so
    results are never silently overwritten.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}_{run_name}" if run_name else stamp
    run_dir = optimisation.results_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    result.schedule.to_csv(run_dir / "schedule.csv")

    assumptions = {
        "run": {
            "name": run_name,
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
        },
        "battery": asdict(battery),
        "optimisation": asdict(optimisation),
        "terminal_soc_value_eur_per_mwh": result.terminal_soc_value_eur_per_mwh,
        "solver_status": result.solver_status,
        "data": provenance or {},
        "summary": result.summary,
    }
    with open(run_dir / "assumptions.json", "w", encoding="utf-8") as handle:
        json.dump(assumptions, handle, indent=2, sort_keys=True, default=str)

    LOGGER.info("wrote results -> %s", run_dir)
    return run_dir


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve a single-window day-ahead arbitrage LP (perfect foresight)."
    )
    parser.add_argument("--start", required=True, help="first local delivery date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last local delivery date, inclusive")
    parser.add_argument("--battery-config", default=None)
    parser.add_argument("--optimisation-config", default=None)
    parser.add_argument("--soc-initial-mwh", type=float, default=None)
    parser.add_argument("--run-name", default=None, help="suffix for the results folder")
    parser.add_argument("--no-write", action="store_true", help="skip the results output")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    prices = fetch_day_ahead_prices(args.start, args.end)
    battery = load_battery_config(args.battery_config)
    optimisation = load_optimisation_config(args.optimisation_config)
    result = solve_window(
        prices, battery, optimisation, soc_initial_mwh=args.soc_initial_mwh
    )

    summary = result.summary
    print(
        f"\n{summary['n_intervals']} intervals ({summary['span_hours']:g} h), "
        f"mean price {summary['mean_price_eur_mwh']:.2f} EUR/MWh"
    )
    print(f"  gross revenue     {summary['gross_revenue_eur']:12,.2f} EUR")
    print(f"  charge cost       {-summary['charge_cost_eur']:12,.2f} EUR")
    print(f"  degradation       {-summary['degradation_cost_eur']:12,.2f} EUR")
    print(f"  net profit        {summary['net_profit_eur']:12,.2f} EUR")
    print(f"  per MW per year   {summary['net_profit_eur_per_mw_year']:12,.2f} EUR/MW/yr")
    print(
        f"  {summary['equivalent_full_cycles']:.1f} equivalent full cycles "
        f"({summary['cycles_per_day']:.2f}/day), closing SoC "
        f"{summary['soc_final_mwh']:.2f} MWh"
    )

    if not args.no_write:
        provenance = {
            "zone_range": f"{args.start}..{args.end}",
            "source": "ENTSO-E day-ahead via src.data",
            "foresight": "perfect (single window)",
        }
        write_run_outputs(
            result, battery, optimisation, provenance, run_name=args.run_name
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
