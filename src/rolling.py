"""Rolling-horizon driver respecting day-ahead gate closure.

The single-window model in :mod:`src.model` sees the whole price series at
once. A real operator does not: at 12:00 on D-1 they commit *every* MTU of
delivery day D in one shot, and the state of charge they open day D with is
whatever the previous gate's committed schedule left them. They cannot revisit
D-1's evening once D's prices clear at 12:45.

This module imposes that structure:

* **Gate** -- 12:00 local on D-1. Local, so 11:00Z in winter and 10:00Z under
  summer time; the recorded gate timestamps reflect the clock change.
* **Commit** -- every interval of local delivery day D. 23, 24 or 25 hours, or
  92/96/100 quarter-hours. Never assumed.
* **Optimise** -- the committed block plus a lookahead into D+1.
* **Discard** -- the lookahead tail. It exists only so the closing SoC is
  chosen against what comes next rather than against the edge of the problem.
* **Carry** -- SoC at the end of day D becomes the next block's opening SoC.
  A hard link: no re-optimisation, no hindsight.

What the gate does and does not constrain
-----------------------------------------
It is not a constraint on price knowledge. A battery bids a price-quantity
curve into the auction and settles at the clearing price; a curve that charges
below X and discharges above Y is self-selecting, so for a price-taking unit
the within-day outcome sits close to what foresight on the cleared prices
gives. What the gate constrains is *structural*: one decision instant per day,
and an inherited state of charge. That is what this module models, which makes
the result a day-ahead arbitrage benchmark rather than a forecast backtest.

The lookahead prices are the realised D+1 prices by default, which were not
known at the gate. That is a bounded optimism -- the tail is discarded, so it
only informs the end-of-day SoC target, and the alternative (a flat proxy)
empties the battery too eagerly each evening and understates. Set
``lookahead_price_source: flat`` in ``config/rolling.yaml`` to measure the
difference.

Usage
-----
    py -m src.rolling --start 2023-01-01 --end 2024-12-31 --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import yaml

from src.data import INDEX_NAME, PRICE_COL, PriceDataError, fetch_day_ahead_prices
from src.model import (
    BatteryConfig,
    OptimisationConfig,
    WindowResult,
    _git_commit,
    derive_terminal_value,
    load_battery_config,
    load_optimisation_config,
    solve_window,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLING_CONFIG = PROJECT_ROOT / "config" / "rolling.yaml"


@dataclass(frozen=True)
class RollingConfig:
    """How the horizon is chopped into commitment blocks."""

    gate_closure_local_time: str = "12:00"
    commitment_days: int = 1
    lookahead_hours: float = 24.0
    lookahead_price_source: str = "realised"
    results_dir: str = "results"

    def __post_init__(self) -> None:
        if self.commitment_days < 1:
            raise ValueError("commitment_days must be at least 1")
        if self.lookahead_hours < 0:
            raise ValueError("lookahead_hours cannot be negative")
        if self.lookahead_price_source not in {"realised", "flat"}:
            raise ValueError("lookahead_price_source must be 'realised' or 'flat'")
        hour, _, minute = self.gate_closure_local_time.partition(":")
        if not (hour.isdigit() and minute.isdigit()):
            raise ValueError("gate_closure_local_time must look like 'HH:MM'")

    @property
    def gate_hour_minute(self) -> tuple:
        hour, _, minute = self.gate_closure_local_time.partition(":")
        return int(hour), int(minute)

    @property
    def results_root(self) -> Path:
        path = Path(self.results_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_rolling_config(path: Optional[Union[str, Path]] = None, **overrides) -> RollingConfig:
    """Load rolling-horizon settings from YAML."""
    config_path = Path(path) if path is not None else DEFAULT_ROLLING_CONFIG
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw.update(overrides)
    return RollingConfig(**raw)


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------

def gate_closure_utc(
    first_delivery_day: pd.Timestamp, tz: str, rolling: RollingConfig
) -> pd.Timestamp:
    """Instant the block is committed: gate time local on the day before D.

    Returned in UTC, so the summer/winter offset is explicit rather than
    implied. This is the timestamp a forecast-driven variant would cut its
    information set at.
    """
    hour, minute = rolling.gate_hour_minute
    local_gate = (
        pd.Timestamp(first_delivery_day).normalize()
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=hour, minutes=minute)
    )
    return local_gate.tz_localize(
        tz, ambiguous=True, nonexistent="shift_forward"
    ).tz_convert("UTC")


@dataclass
class Block:
    """One commitment block: what is committed, and what is only looked at."""

    index: int
    delivery_days: list
    commit_start: int          # positional slice into the price frame
    commit_stop: int
    window_stop: int           # commit_stop plus the lookahead tail
    gate_closure_utc: pd.Timestamp

    @property
    def n_committed(self) -> int:
        return self.commit_stop - self.commit_start

    @property
    def n_lookahead(self) -> int:
        return self.window_stop - self.commit_stop


def build_blocks(
    prices: pd.DataFrame,
    tz: str,
    rolling: RollingConfig,
    horizon_days: Optional[set] = None,
) -> list:
    """Partition the price frame into commitment blocks with lookahead tails.

    Blocks are cut on local delivery days, so their length follows the clock
    rather than a fixed interval count.

    ``prices`` may extend past the horizon so the final block still has a tail
    to look into; ``horizon_days`` then names the days actually committed.
    """
    if "delivery_date_local" not in prices.columns:
        raise ValueError(
            "price frame needs a 'delivery_date_local' column; "
            "use src.data.fetch_day_ahead_prices"
        )

    all_days = list(pd.unique(prices["delivery_date_local"]))
    if horizon_days is None:
        days = all_days
    else:
        wanted = {pd.Timestamp(d) for d in horizon_days}
        days = [d for d in all_days if pd.Timestamp(d) in wanted]
    if not days:
        raise ValueError("no delivery days to commit")

    day_of = prices["delivery_date_local"].to_numpy()
    hours = prices["interval_hours"].to_numpy(dtype="float64")
    n = len(prices)

    blocks = []
    for block_index, position in enumerate(range(0, len(days), rolling.commitment_days)):
        block_days = days[position : position + rolling.commitment_days]
        in_block = np.isin(day_of, block_days)
        positions = np.flatnonzero(in_block)
        commit_start, commit_stop = int(positions[0]), int(positions[-1]) + 1

        # Walk forward until the lookahead budget is spent or data runs out.
        window_stop = commit_stop
        accumulated = 0.0
        while window_stop < n and accumulated < rolling.lookahead_hours - 1e-9:
            accumulated += hours[window_stop]
            window_stop += 1

        blocks.append(
            Block(
                index=block_index,
                delivery_days=[pd.Timestamp(d) for d in block_days],
                commit_start=commit_start,
                commit_stop=commit_stop,
                window_stop=window_stop,
                gate_closure_utc=gate_closure_utc(block_days[0], tz, rolling),
            )
        )
    return blocks


def _window_prices(prices: pd.DataFrame, block: Block, rolling: RollingConfig) -> pd.DataFrame:
    """Prices for one optimisation window, with the lookahead tail treated."""
    window = prices.iloc[block.commit_start : block.window_stop].copy()
    if rolling.lookahead_price_source == "flat" and block.n_lookahead:
        committed_mean = float(window[PRICE_COL].iloc[: block.n_committed].mean())
        window.iloc[block.n_committed :, window.columns.get_loc(PRICE_COL)] = committed_mean
    return window


# ---------------------------------------------------------------------------
# Rolling solve
# ---------------------------------------------------------------------------

@dataclass
class RollingResult:
    """Committed schedule across all blocks, plus per-block accounting."""

    schedule: pd.DataFrame
    blocks: pd.DataFrame
    summary: dict


def run_rolling(
    prices: pd.DataFrame,
    battery: Optional[BatteryConfig] = None,
    optimisation: Optional[OptimisationConfig] = None,
    rolling: Optional[RollingConfig] = None,
    timezone_name: str = "Europe/Berlin",
    soc_initial_mwh: Optional[float] = None,
    horizon_days: Optional[set] = None,
    progress_every: int = 100,
) -> RollingResult:
    """Solve the horizon block by block, committing only day D each time.

    ``prices`` may run past the horizon so the last block has a lookahead tail;
    ``horizon_days`` then names the delivery days actually committed.
    """
    battery = battery or load_battery_config()
    optimisation = optimisation or load_optimisation_config()
    rolling = rolling or load_rolling_config()

    blocks = build_blocks(prices, timezone_name, rolling, horizon_days)
    soc = battery.soc_initial_mwh if soc_initial_mwh is None else float(soc_initial_mwh)

    LOGGER.info(
        "rolling horizon: %d blocks, %s commitment day(s) each, %g h lookahead (%s prices)",
        len(blocks), rolling.commitment_days, rolling.lookahead_hours,
        rolling.lookahead_price_source,
    )

    committed_parts = []
    block_rows = []

    for block in blocks:
        window = _window_prices(prices, block, rolling)
        result = solve_window(
            window, battery, optimisation, soc_initial_mwh=soc
        )

        # Keep only what the gate actually commits; the tail is discarded.
        committed = result.schedule.iloc[: block.n_committed].copy()
        committed["block_index"] = block.index
        committed["gate_closure_utc"] = block.gate_closure_utc

        # The gate must precede delivery, or the block is not a commitment.
        if block.gate_closure_utc >= committed.index[0]:
            raise ValueError(
                f"block {block.index} gate {block.gate_closure_utc} does not precede "
                f"first committed interval {committed.index[0]}"
            )

        soc_open, soc = soc, float(committed["soc_end_mwh"].iloc[-1])
        committed_parts.append(committed)

        block_rows.append(
            {
                "block_index": block.index,
                "delivery_date_local": block.delivery_days[0],
                "gate_closure_utc": block.gate_closure_utc,
                "n_committed": block.n_committed,
                "n_lookahead": block.n_lookahead,
                "committed_hours": float(committed["interval_hours"].sum()),
                "soc_open_mwh": soc_open,
                "soc_close_mwh": soc,
                "mean_price_eur_mwh": float(committed[PRICE_COL].mean()),
                "energy_charged_mwh": float(committed["energy_charged_mwh"].sum()),
                "energy_discharged_mwh": float(committed["energy_discharged_mwh"].sum()),
                "gross_revenue_eur": float(committed["discharge_revenue_eur"].sum()),
                "charge_cost_eur": float(committed["charge_cost_eur"].sum()),
                "degradation_cost_eur": float(committed["degradation_cost_eur"].sum()),
                "net_eur": float(committed["net_eur"].sum()),
                # Trading profit alone misreads a gate that deliberately ends
                # holding stock: the purchase lands in this block, the sale in
                # the next. Value the inventory change at the rate the
                # optimiser itself used, so each gate's P&L stands alone.
                "inventory_change_mwh": soc - soc_open,
                "inventory_value_change_eur": (
                    result.terminal_soc_value_eur_per_mwh * (soc - soc_open)
                ),
                "net_incl_inventory_eur": (
                    float(committed["net_eur"].sum())
                    + result.terminal_soc_value_eur_per_mwh * (soc - soc_open)
                ),
                "terminal_soc_value_eur_per_mwh": result.terminal_soc_value_eur_per_mwh,
                "solver_seconds": result.solver_seconds,
            }
        )

        if progress_every and (block.index + 1) % progress_every == 0:
            done = sum(row["net_eur"] for row in block_rows)
            LOGGER.info(
                "  gate %d/%d (%s): cumulative net %.0f EUR",
                block.index + 1, len(blocks),
                block.delivery_days[0].date(), done,
            )

    schedule = pd.concat(committed_parts)
    schedule.index.name = INDEX_NAME
    blocks_df = pd.DataFrame(block_rows).set_index("block_index")

    # The committed blocks must tile the committed days exactly once: no
    # interval optimised twice, none dropped between gates.
    committed_days = {pd.Timestamp(d) for b in blocks for d in b.delivery_days}
    expected = prices.loc[
        prices["delivery_date_local"].isin(committed_days)
    ]
    if not schedule.index.equals(expected.index):
        raise ValueError(
            f"committed {len(schedule)} intervals but the committed days hold "
            f"{len(expected)}; blocks do not tile the horizon"
        )

    # SoC must chain across gates with no hidden reset.
    opens = blocks_df["soc_open_mwh"].to_numpy()
    closes = blocks_df["soc_close_mwh"].to_numpy()
    if len(opens) > 1:
        drift = float(np.abs(opens[1:] - closes[:-1]).max())
        if drift > 1e-9:
            raise ValueError(f"state of charge breaks continuity across gates by {drift:g} MWh")

    summary = _summarise_rolling(schedule, blocks_df, battery, rolling)
    LOGGER.info(
        "rolling complete: net %.2f EUR over %d blocks, %.1f EFC",
        summary["net_profit_eur"], len(blocks_df), summary["equivalent_full_cycles"],
    )
    return RollingResult(schedule=schedule, blocks=blocks_df, summary=summary)


def _summarise_rolling(
    schedule: pd.DataFrame,
    blocks: pd.DataFrame,
    battery: BatteryConfig,
    rolling: RollingConfig,
) -> dict:
    discharged = float(schedule["energy_discharged_mwh"].sum())
    net = float(schedule["net_eur"].sum())
    span_hours = float(schedule["interval_hours"].sum())

    return {
        "n_blocks": int(len(blocks)),
        "n_intervals": int(len(schedule)),
        "span_hours": span_hours,
        "first_interval_utc": schedule.index[0].isoformat(),
        "last_interval_utc": schedule.index[-1].isoformat(),
        "commitment_days": rolling.commitment_days,
        "lookahead_hours": rolling.lookahead_hours,
        "lookahead_price_source": rolling.lookahead_price_source,
        "mean_price_eur_mwh": float(schedule[PRICE_COL].mean()),
        "energy_charged_mwh": float(schedule["energy_charged_mwh"].sum()),
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
        "soc_initial_mwh": float(blocks["soc_open_mwh"].iloc[0]),
        "soc_final_mwh": float(blocks["soc_close_mwh"].iloc[-1]),
        "best_block_net_eur": float(blocks["net_incl_inventory_eur"].max()),
        "worst_block_net_eur": float(blocks["net_incl_inventory_eur"].min()),
        "loss_making_blocks": int((blocks["net_incl_inventory_eur"] < -1e-6).sum()),
        "best_block_date": str(
            blocks.loc[blocks["net_incl_inventory_eur"].idxmax(), "delivery_date_local"].date()
        ),
        "worst_block_date": str(
            blocks.loc[blocks["net_incl_inventory_eur"].idxmin(), "delivery_date_local"].date()
        ),
        "solver_seconds_total": float(blocks["solver_seconds"].sum()),
    }


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def compare_to_perfect_foresight(
    rolling_result: RollingResult,
    horizon: pd.DataFrame,
    battery: BatteryConfig,
    optimisation: OptimisationConfig,
    soc_initial_mwh: Optional[float] = None,
) -> dict:
    """Measure what the gate-closure structure costs, on a like-for-like basis.

    Trading profit alone is **not** comparable between the two runs. Both start
    at the same SoC but they do not end at one: perfect foresight typically
    finishes holding a full battery it bought inside the horizon, while a
    rolling run that sees a cheap tomorrow sells its stock. Comparing raw
    trading profit therefore credits the run that liquidated and can make the
    rolling result look like it *beat* full foresight, which is impossible --
    perfect foresight optimises over a superset of the information and its
    schedule is feasible for any block structure.

    So both closing inventories are marked at one common rate, derived from the
    whole horizon. The comparison is then between two runs with identical
    opening and equivalently-valued closing positions.
    """
    perfect = solve_window(
        horizon, battery, optimisation, soc_initial_mwh=soc_initial_mwh
    )
    inventory_rate = derive_terminal_value(horizon, battery)

    rolling_net = rolling_result.summary["net_profit_eur"]
    rolling_soc = rolling_result.summary["soc_final_mwh"]
    perfect_net = perfect.summary["net_profit_eur"]
    perfect_soc = perfect.summary["soc_final_mwh"]

    rolling_total = rolling_net + inventory_rate * rolling_soc
    perfect_total = perfect_net + inventory_rate * perfect_soc

    if rolling_total > perfect_total + 1e-6:
        raise ValueError(
            f"rolling horizon ({rolling_total:,.2f} EUR) beat perfect foresight "
            f"({perfect_total:,.2f} EUR) on a like-for-like basis, which cannot "
            "happen -- the benchmark or the block structure is wrong"
        )

    capture = rolling_total / perfect_total if perfect_total else float("nan")
    return {
        "inventory_rate_eur_per_mwh": inventory_rate,
        "rolling_trading_eur": rolling_net,
        "rolling_closing_soc_mwh": rolling_soc,
        "rolling_total_eur": rolling_total,
        "perfect_trading_eur": perfect_net,
        "perfect_closing_soc_mwh": perfect_soc,
        "perfect_total_eur": perfect_total,
        "gate_closure_cost_eur": perfect_total - rolling_total,
        "capture_rate": capture,
        "perfect_foresight_summary": perfect.summary,
    }


# ---------------------------------------------------------------------------
# Horizon data
# ---------------------------------------------------------------------------

def load_horizon_prices(
    start: str, end: str, rolling: RollingConfig, **fetch_kwargs
) -> pd.DataFrame:
    """Fetch the horizon plus enough extra days to feed the final lookahead.

    Falls back to the bare horizon when the extension is not published yet, so
    the last block simply optimises without a tail.
    """
    extra_days = math.ceil(rolling.lookahead_hours / 24.0)
    if extra_days <= 0:
        return fetch_day_ahead_prices(start, end, **fetch_kwargs)

    extended_end = (pd.Timestamp(end) + pd.Timedelta(days=extra_days)).date()
    try:
        return fetch_day_ahead_prices(start, str(extended_end), **fetch_kwargs)
    except PriceDataError as exc:
        LOGGER.warning(
            "lookahead extension to %s unavailable (%s); the final block will "
            "optimise without a tail",
            extended_end, exc,
        )
        return fetch_day_ahead_prices(start, end, **fetch_kwargs)


def horizon_slice(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Trim an extended frame back to the delivery days actually being run."""
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    within = (prices["delivery_date_local"] >= first) & (
        prices["delivery_date_local"] <= last
    )
    return prices.loc[within]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_rolling_outputs(
    result: RollingResult,
    battery: BatteryConfig,
    optimisation: OptimisationConfig,
    rolling: RollingConfig,
    provenance: Optional[dict] = None,
    run_name: Optional[str] = None,
    benchmark: Optional[dict] = None,
) -> Path:
    """Write the committed schedule, per-block ledger and assumptions dump."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}_{run_name}" if run_name else f"{stamp}_rolling"
    run_dir = rolling.results_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    result.schedule.to_csv(run_dir / "schedule.csv")
    result.blocks.to_csv(run_dir / "blocks.csv")

    assumptions = {
        "run": {
            "name": run_name,
            "kind": "rolling horizon",
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
        },
        "battery": asdict(battery),
        "optimisation": asdict(optimisation),
        "rolling": asdict(rolling),
        "data": provenance or {},
        "summary": result.summary,
        "perfect_foresight_benchmark": benchmark or {},
    }
    with open(run_dir / "assumptions.json", "w", encoding="utf-8") as handle:
        json.dump(assumptions, handle, indent=2, sort_keys=True, default=str)

    LOGGER.info("wrote results -> %s", run_dir)
    return run_dir


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the gate-closure rolling horizon over a date range."
    )
    parser.add_argument("--start", required=True, help="first local delivery date")
    parser.add_argument("--end", required=True, help="last local delivery date, inclusive")
    parser.add_argument("--battery-config", default=None)
    parser.add_argument("--optimisation-config", default=None)
    parser.add_argument("--rolling-config", default=None)
    parser.add_argument("--soc-initial-mwh", type=float, default=None)
    parser.add_argument(
        "--lookahead-price-source", choices=["realised", "flat"], default=None
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="also solve the whole range as one perfect-foresight window",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # 700+ solves would otherwise emit 700+ lines from the window solver.
    logging.getLogger("src.model").setLevel(logging.WARNING)
    logging.getLogger("pyomo").setLevel(logging.WARNING)

    overrides = {}
    if args.lookahead_price_source:
        overrides["lookahead_price_source"] = args.lookahead_price_source
    rolling = load_rolling_config(args.rolling_config, **overrides)
    battery = load_battery_config(args.battery_config)
    optimisation = load_optimisation_config(args.optimisation_config)

    # Fetch past the horizon so the final gate still has something to look at,
    # then commit only the horizon's own delivery days.
    extended = load_horizon_prices(args.start, args.end, rolling)
    horizon = horizon_slice(extended, args.start, args.end)
    horizon_days = set(pd.unique(horizon["delivery_date_local"]))
    window_source = extended.loc[extended.index >= horizon.index[0]]

    result = run_rolling(
        window_source,
        battery,
        optimisation,
        rolling,
        soc_initial_mwh=args.soc_initial_mwh,
        horizon_days=horizon_days,
    )

    summary = result.summary
    print(
        f"\nrolling horizon: {summary['n_blocks']} gates, {summary['n_intervals']} "
        f"intervals ({summary['span_hours']:g} h)"
    )
    print(f"  net profit        {summary['net_profit_eur']:12,.2f} EUR")
    print(f"  per MW per year   {summary['net_profit_eur_per_mw_year']:12,.2f} EUR/MW/yr")
    print(
        f"  {summary['equivalent_full_cycles']:.1f} equivalent full cycles "
        f"({summary['cycles_per_day']:.2f}/day)"
    )
    print(
        f"  worst gate {summary['worst_block_net_eur']:,.2f} EUR, "
        f"{summary['loss_making_blocks']} loss-making gate(s)"
    )

    benchmark = None
    if args.compare:
        LOGGER.info("solving the same range as a single perfect-foresight window")
        benchmark = compare_to_perfect_foresight(
            result, horizon, battery, optimisation, args.soc_initial_mwh
        )
        print(
            f"\n  closing inventory marked at "
            f"{benchmark['inventory_rate_eur_per_mwh']:.2f} EUR/MWh"
        )
        print(
            f"  rolling           {benchmark['rolling_total_eur']:12,.2f} EUR "
            f"(trading {benchmark['rolling_trading_eur']:,.0f} + "
            f"{benchmark['rolling_closing_soc_mwh']:.1f} MWh held)"
        )
        print(
            f"  perfect foresight {benchmark['perfect_total_eur']:12,.2f} EUR "
            f"(trading {benchmark['perfect_trading_eur']:,.0f} + "
            f"{benchmark['perfect_closing_soc_mwh']:.1f} MWh held)"
        )
        print(
            f"  cost of gate closure {benchmark['gate_closure_cost_eur']:12,.2f} EUR"
        )
        print(f"  capture rate      {100*benchmark['capture_rate']:12.2f}%")

    if not args.no_write:
        provenance = {
            "zone_range": f"{args.start}..{args.end}",
            "source": "ENTSO-E day-ahead via src.data",
            "foresight": f"block commitment, {rolling.lookahead_price_source} lookahead",
        }
        write_rolling_outputs(
            result, battery, optimisation, rolling, provenance,
            run_name=args.run_name, benchmark=benchmark,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
