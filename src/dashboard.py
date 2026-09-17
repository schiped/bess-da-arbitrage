"""Build the self-contained results dashboard.

Runs the rolling horizon and its perfect-foresight benchmark over a date range,
assembles everything the page needs into one JSON payload, and injects it into
``dashboard/template.html`` to produce a single standalone HTML file -- no
network, no build step, no external data.

The lookahead sensitivity sweep re-runs the whole horizon once per setting, so
it is the expensive part; ``--quick`` skips it.

Usage
-----
    py -m src.dashboard --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.data import PRICE_COL, fetch_day_ahead_prices
from src.model import (
    load_battery_config,
    load_optimisation_config,
    solve_window,
)
from src.rolling import (
    compare_to_perfect_foresight,
    horizon_slice,
    load_rolling_config,
    run_rolling,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = PROJECT_ROOT / "dashboard" / "template.html"
DATA_MARKER = "/*__DATA__*/"

#: Lookahead settings swept for the sensitivity panel: (hours, price source).
SENSITIVITY_GRID = [
    (0, "realised"),
    (6, "realised"),
    (12, "realised"),
    (24, "realised"),
    (36, "realised"),
    (48, "realised"),
    (24, "flat"),
]


def _dispatch_rows(schedule: pd.DataFrame, day: str, timezone_name: str) -> list:
    """Per-interval dispatch for one delivery day, with the local clock kept."""
    rows = schedule.loc[schedule["delivery_date_local"] == pd.Timestamp(day)]
    out = []
    for stamp, row in rows.iterrows():
        local = stamp.tz_convert(timezone_name)
        out.append(
            {
                # Keep the UTC offset: on a 25-hour day the two 02:00 intervals
                # are only told apart by it.
                "local": local.strftime("%H:%M"),
                "offset": local.strftime("%z"),
                "price": round(float(row[PRICE_COL]), 2),
                "charge": round(float(row["charge_mw"]), 3),
                "discharge": round(float(row["discharge_mw"]), 3),
                "soc": round(float(row["soc_end_mwh"]), 3),
            }
        )
    return out


def build_payload(
    start: str,
    end: str,
    *,
    quick: bool = False,
    timezone_name: str = "Europe/Berlin",
) -> dict:
    """Run every case the dashboard reports and collect it into one dict."""
    battery = load_battery_config()
    optimisation = load_optimisation_config()
    base_rolling = load_rolling_config()

    extra = pd.Timestamp(end) + pd.Timedelta(days=2)
    extended = fetch_day_ahead_prices(start, str(extra.date()))
    horizon = horizon_slice(extended, start, end)
    horizon_days = set(pd.unique(horizon["delivery_date_local"]))
    window_source = extended.loc[extended.index >= horizon.index[0]]

    LOGGER.info("baseline rolling run")
    base = run_rolling(
        window_source, battery, optimisation, base_rolling,
        horizon_days=horizon_days, progress_every=0,
    )
    LOGGER.info("perfect-foresight benchmark")
    comparison = compare_to_perfect_foresight(base, horizon, battery, optimisation)
    perfect = solve_window(horizon, battery, optimisation)
    rate = comparison["inventory_rate_eur_per_mwh"]
    ceiling = comparison["perfect_total_eur"]

    sensitivity = []
    grid = [(base_rolling.lookahead_hours, base_rolling.lookahead_price_source)] \
        if quick else SENSITIVITY_GRID
    for hours, source in grid:
        LOGGER.info("sensitivity: lookahead=%s source=%s", hours, source)
        config = load_rolling_config(
            lookahead_hours=hours, lookahead_price_source=source
        )
        run = run_rolling(
            window_source, battery, optimisation, config,
            horizon_days=horizon_days, progress_every=0,
        )
        total = run.summary["net_profit_eur"] + rate * run.summary["soc_final_mwh"]
        sensitivity.append(
            {
                "lookahead_hours": hours,
                "source": source,
                "trading_eur": run.summary["net_profit_eur"],
                "total_eur": total,
                "capture": total / ceiling,
                "cycles": run.summary["equivalent_full_cycles"],
                "eur_per_mw_year": run.summary["net_profit_eur_per_mw_year"],
            }
        )

    # ---- daily and monthly series ----
    blocks = base.blocks.copy()
    blocks["date"] = blocks["delivery_date_local"].dt.strftime("%Y-%m-%d")
    price_by_day = horizon.groupby("delivery_date_local")[PRICE_COL].agg(["mean", "min", "max"])
    price_by_day["spread"] = price_by_day["max"] - price_by_day["min"]
    price_by_day = price_by_day.reset_index()
    price_by_day["date"] = price_by_day["delivery_date_local"].dt.strftime("%Y-%m-%d")

    merged = blocks.merge(price_by_day[["date", "mean", "spread"]], on="date", how="left")
    merged["cumulative_eur"] = merged["net_incl_inventory_eur"].cumsum()

    perfect_daily = (
        perfect.schedule.groupby("delivery_date_local")["net_eur"].sum().cumsum()
    )

    daily = [
        {
            "date": row["date"],
            "net": round(float(row["net_incl_inventory_eur"]), 2),
            "cum": round(float(row["cumulative_eur"]), 2),
            "spread": round(float(row["spread"]), 2),
            "mean_price": round(float(row["mean"]), 2),
            "cycles": round(
                float(row["energy_discharged_mwh"]) / battery.energy_capacity_mwh, 4
            ),
            "hours": int(row["committed_hours"]),
        }
        for _, row in merged.iterrows()
    ]

    merged["month"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m")
    by_month = merged.groupby("month").agg(
        net=("net_incl_inventory_eur", "sum"),
        cycles=("energy_discharged_mwh", lambda s: s.sum() / battery.energy_capacity_mwh),
        spread=("spread", "mean"),
    ).reset_index()

    # ---- clock changes ----
    day_lengths = horizon.groupby("delivery_date_local")["interval_hours"].sum()
    dst_days = []
    for day, hours in day_lengths[day_lengths != 24.0].items():
        row = merged[merged["date"] == day.strftime("%Y-%m-%d")].iloc[0]
        dst_days.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "hours": float(hours),
                "intervals": int(row["n_committed"]),
                "net": round(float(row["net_incl_inventory_eur"]), 2),
                "gate_utc": pd.Timestamp(row["gate_closure_utc"]).strftime(
                    "%Y-%m-%d %H:%MZ"
                ),
            }
        )

    best = merged.loc[merged["net_incl_inventory_eur"].idxmax()]
    longest_day = day_lengths.idxmax()

    return {
        "meta": {
            "zone": "DE-LU",
            "start": start,
            "end": end,
            "intervals": int(len(horizon)),
            "days": int(len(merged)),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        },
        "battery": {
            "power_mw": battery.power_charge_max_mw,
            "energy_mwh": battery.energy_capacity_mwh,
            "eta_charge": battery.efficiency_charge,
            "eta_discharge": battery.efficiency_discharge,
            "round_trip": round(battery.round_trip_efficiency, 4),
            "degradation": battery.degradation_eur_per_mwh_discharged,
            "in_place_threshold": round(battery.in_place_cycling_price_threshold, 2),
        },
        "headline": {
            "rolling_total": comparison["rolling_total_eur"],
            "rolling_trading": comparison["rolling_trading_eur"],
            "perfect_total": comparison["perfect_total_eur"],
            "perfect_trading": comparison["perfect_trading_eur"],
            "gate_cost": comparison["gate_closure_cost_eur"],
            "capture": comparison["capture_rate"],
            "eur_per_mw_year": base.summary["net_profit_eur_per_mw_year"],
            "cycles": base.summary["equivalent_full_cycles"],
            "cycles_per_day": base.summary["cycles_per_day"],
            "gross_revenue": base.summary["gross_revenue_eur"],
            "charge_cost": base.summary["charge_cost_eur"],
            "degradation_cost": base.summary["degradation_cost_eur"],
            "energy_charged": base.summary["energy_charged_mwh"],
            "energy_discharged": base.summary["energy_discharged_mwh"],
            "mean_price": base.summary["mean_price_eur_mwh"],
            "n_gates": base.summary["n_blocks"],
            "best_day": best["date"],
            "best_day_net": round(float(best["net_incl_inventory_eur"]), 2),
            "loss_making_gates": base.summary["loss_making_blocks"],
            "solver_seconds": base.summary["solver_seconds_total"],
            "exclusivity_binaries": int(perfect.summary.get("exclusivity_binaries", 0)),
            "min_price": round(float(horizon[PRICE_COL].min()), 2),
            "max_price": round(float(horizon[PRICE_COL].max()), 2),
            "negative_hours": int((horizon[PRICE_COL] < 0).sum()),
        },
        "sensitivity": sensitivity,
        "daily": daily,
        "perfect_cumulative": [round(float(v), 2) for v in perfect_daily],
        "monthly": [
            {
                "month": r["month"],
                "net": round(float(r["net"]), 2),
                "cycles": round(float(r["cycles"]), 2),
                "spread": round(float(r["spread"]), 2),
            }
            for _, r in by_month.iterrows()
        ],
        "dispatch_dst": _dispatch_rows(
            base.schedule, longest_day.strftime("%Y-%m-%d"), timezone_name
        ),
        "dispatch_best": _dispatch_rows(base.schedule, best["date"], timezone_name),
        "dst_days": dst_days,
    }


def render(payload: dict, output_path: Path, template_path: Path = TEMPLATE_PATH) -> Path:
    """Inject the payload into the template and write a standalone HTML file."""
    template = template_path.read_text(encoding="utf-8")
    if DATA_MARKER not in template:
        raise ValueError(f"{template_path} has no {DATA_MARKER} placeholder")
    page = template.replace(
        DATA_MARKER, json.dumps(payload, separators=(",", ":"), default=str)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the results dashboard.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--output", default=None,
        help="output HTML path (default: dashboard/index.html)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="skip the lookahead sweep -- one horizon run instead of seven",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("src.model").setLevel(logging.WARNING)
    logging.getLogger("src.rolling").setLevel(logging.WARNING)
    logging.getLogger("pyomo").setLevel(logging.WARNING)

    payload = build_payload(args.start, args.end, quick=args.quick)
    output = Path(args.output) if args.output else PROJECT_ROOT / "dashboard" / "index.html"
    path = render(payload, output)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path} ({size_kb:,.0f} KB, {len(payload['daily'])} delivery days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
