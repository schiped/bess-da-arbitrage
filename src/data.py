"""Day-ahead price ingestion for the ENTSO-E Transparency Platform.

Fetches day-ahead (SDAC) prices for a bidding zone over a range of *local
delivery dates*, caches the result to parquet under ``data/raw`` keyed by zone
and range, and returns a DataFrame indexed by UTC timestamp.

Time handling
-------------
The day-ahead market is settled on local calendar days, but the model needs a
monotonic, gap-free, duplicate-free time axis. Those two facts only reconcile
in UTC, so this module is deliberate about the boundary between them:

* ``start`` / ``end`` are **local delivery dates**, inclusive on both ends.
  "2024-10-27" means the German delivery day, whatever length it happens to be.
* The request window is built by localising local midnight to the zone
  timezone, which picks up the correct UTC offset on either side of a clock
  change. The window is half-open in UTC: ``[utc_start, utc_end)``.
* The returned index is tz-aware UTC and therefore strictly increasing and
  unique across both transitions.

Consequences worth stating, because they are what breaks naive pipelines:

* The **last Sunday in October** is a 25-hour local day -> 25 hourly intervals
  (100 at quarter-hourly resolution). Local wall-clock time repeats 02:00
  twice; in UTC the two intervals are distinct.
* The **last Sunday in March** is a 23-hour local day -> 23 hourly intervals
  (92 at quarter-hourly). Local 02:00 does not exist at all that day.
* An N-day range therefore does **not** contain ``24 * N`` intervals. Nothing
  here assumes it does; completeness is validated against the elapsed UTC span.

Resolution
----------
SDAC day-ahead was hourly until 2025-10-01 and is quarter-hourly from that date
(entsoe-py forces the correct one). A range straddling the switch contains both,
so every interval carries its own ``interval_hours`` rather than relying on a
single global step. Multiply power in MW by ``interval_hours`` for energy in
MWh; never assume 1.0.

Usage
-----
    from src.data import fetch_day_ahead_prices
    prices = fetch_day_ahead_prices("2023-01-01", "2024-12-31")

or, to warm the cache from the shell::

    py -m src.data --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import yaml
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"

#: Column carrying the cleared day-ahead price.
PRICE_COL = "price_eur_mwh"
#: Name of the returned (UTC, tz-aware) index.
INDEX_NAME = "timestamp_utc"

DateLike = Union[str, pd.Timestamp, datetime]


class PriceDataError(RuntimeError):
    """Raised when fetched prices fail the completeness/consistency checks."""


@dataclass(frozen=True)
class DataConfig:
    """Ingestion settings, loaded from ``config/data.yaml``."""

    zone: str
    timezone: str
    cache_dir: Path
    api_key_env: str
    allowed_resolutions_minutes: tuple = (15, 30, 60)
    strict: bool = True
    max_request_days: int = 180
    request_overlap_days: int = 1

    @property
    def cache_root(self) -> Path:
        """Cache directory resolved against the project root."""
        path = Path(self.cache_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_data_config(path: Optional[Union[str, Path]] = None, **overrides) -> DataConfig:
    """Load the ingestion config, applying any keyword overrides on top."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw.update(overrides)
    return DataConfig(
        zone=raw["zone"],
        timezone=raw["timezone"],
        cache_dir=Path(raw["cache_dir"]),
        api_key_env=raw["api_key_env"],
        allowed_resolutions_minutes=tuple(raw.get("allowed_resolutions_minutes", (15, 30, 60))),
        strict=bool(raw.get("strict", True)),
        max_request_days=int(raw.get("max_request_days", 180)),
        request_overlap_days=int(raw.get("request_overlap_days", 1)),
    )


# ---------------------------------------------------------------------------
# Date / DST handling
# ---------------------------------------------------------------------------

def _as_delivery_date(value: DateLike) -> pd.Timestamp:
    """Normalise user input to a naive, midnight-aligned local delivery date."""
    stamp = pd.Timestamp(value)
    if stamp.tz is not None:
        # An explicit instant is taken as the delivery day it falls on.
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _localise_midnight(day: pd.Timestamp, tz: str) -> pd.Timestamp:
    """Localise a naive local midnight, tolerating clock changes.

    EU transitions happen at 02:00/03:00 local, so midnight is never ambiguous
    or nonexistent in practice -- the explicit arguments are there so a zone
    that does shift at midnight resolves deterministically instead of raising.
    """
    return day.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")


def delivery_window_utc(
    start: DateLike, end: DateLike, tz: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the half-open UTC window ``[utc_start, utc_end)`` for local days.

    ``start`` and ``end`` are inclusive local delivery dates. The span of the
    returned window absorbs any clock change inside the range, which is what
    makes the 23- and 25-hour days fall out correctly instead of needing a
    special case.
    """
    first = _as_delivery_date(start)
    last = _as_delivery_date(end)
    if last < first:
        raise ValueError(f"end date {last.date()} precedes start date {first.date()}")

    utc_start = _localise_midnight(first, tz).tz_convert("UTC")
    # Exclusive upper bound: midnight at the start of the day *after* `end`.
    utc_end = _localise_midnight(last + pd.Timedelta(days=1), tz).tz_convert("UTC")
    return utc_start, utc_end


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_path(start: DateLike, end: DateLike, config: DataConfig) -> Path:
    """Parquet path for a (zone, range) key."""
    first = _as_delivery_date(start).date()
    last = _as_delivery_date(end).date()
    return config.cache_root / f"day_ahead_{config.zone}_{first}_{last}.parquet"


def _metadata_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".meta.json")


def _write_cache(df: pd.DataFrame, parquet_path: Path, metadata: dict) -> None:
    """Write parquet + metadata sidecar atomically, so a kill leaves no stub."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_parquet, engine="pyarrow", index=True)
    os.replace(tmp_parquet, parquet_path)

    meta_path = _metadata_path(parquet_path)
    tmp_meta = meta_path.with_suffix(".json.tmp")
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    os.replace(tmp_meta, meta_path)


def _read_cache(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    # Parquet round-trips tz-aware timestamps, but normalise defensively so a
    # file written by another engine still yields a UTC index.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = INDEX_NAME
    return df


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _build_client(config: DataConfig):
    """Construct an EntsoePandasClient from the API key in the environment."""
    # Imported lazily so a cache hit needs neither the network stack nor a key.
    from entsoe import EntsoePandasClient

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{config.api_key_env} is not set. Put it in .env at the project root "
            "or export it before running."
        )
    return EntsoePandasClient(api_key=api_key)


def _request_chunks(
    utc_start: pd.Timestamp, utc_end: pd.Timestamp, config: DataConfig
) -> list:
    """Split the window into overlapping sub-requests.

    entsoe-py 0.8.0 splits any request longer than a year internally and loses
    the single interval that lands on its split boundary. Chunking below that
    limit here keeps that code path dormant; the overlap means a boundary is
    always covered twice, so a dropped edge from any future library change
    still cannot open a hole.
    """
    span = pd.Timedelta(days=max(1, config.max_request_days))
    overlap = pd.Timedelta(days=max(0, config.request_overlap_days))

    chunks = []
    cursor = utc_start
    while cursor < utc_end:
        chunk_end = min(cursor + span, utc_end)
        chunks.append((max(utc_start - overlap, cursor - overlap), chunk_end + overlap))
        cursor = chunk_end
    return chunks


def _download(
    utc_start: pd.Timestamp, utc_end: pd.Timestamp, config: DataConfig, client=None
) -> pd.Series:
    """Pull raw prices from ENTSO-E for the half-open UTC window."""
    client = client or _build_client(config)
    chunks = _request_chunks(utc_start, utc_end, config)
    LOGGER.info(
        "querying ENTSO-E day-ahead prices: zone=%s window=[%s, %s) in %d request(s)",
        config.zone,
        utc_start,
        utc_end,
        len(chunks),
    )

    parts = []
    for chunk_start, chunk_end in chunks:
        LOGGER.debug("  chunk [%s, %s]", chunk_start, chunk_end)
        # entsoe-py truncates inclusively on `end`, so each chunk returns one
        # interval past its bound; the half-open filter in _build_frame and the
        # dedupe below absorb that.
        part = client.query_day_ahead_prices(config.zone, start=chunk_start, end=chunk_end)
        parts.append(part.tz_convert("UTC"))

    series = pd.concat(parts).sort_index()
    return series[~series.index.duplicated(keep="first")]


def _interval_hours(index: pd.DatetimeIndex) -> pd.Series:
    """Duration of each interval, inferred from the gap to the next one.

    The final interval has no successor, so it inherits the preceding step --
    the resolution only ever changes at a day boundary.
    """
    gaps = index.to_series().diff().dropna()
    if gaps.empty:
        raise PriceDataError("need at least two intervals to infer the resolution")
    durations = list(gaps) + [gaps.iloc[-1]]
    return pd.Series(
        [d.total_seconds() / 3600.0 for d in durations], index=index, name="interval_hours"
    )


def _validate(
    df: pd.DataFrame, utc_start: pd.Timestamp, utc_end: pd.Timestamp, config: DataConfig
) -> None:
    """Check the series is complete, unique and at an expected resolution."""
    if df.empty:
        raise PriceDataError(f"ENTSO-E returned no prices for [{utc_start}, {utc_end})")

    problems: list = []

    if not df.index.is_monotonic_increasing:
        problems.append("index is not monotonically increasing")
    if not df.index.is_unique:
        dupes = df.index[df.index.duplicated()].unique()
        # A duplicated UTC stamp means local times were collapsed somewhere --
        # exactly the October failure mode.
        problems.append(f"{len(dupes)} duplicated UTC timestamps, first at {dupes[0]}")
    if df[PRICE_COL].isna().any():
        problems.append(f"{int(df[PRICE_COL].isna().sum())} missing prices")

    allowed = {round(m / 60.0, 6) for m in config.allowed_resolutions_minutes}
    observed = sorted({round(h, 6) for h in df["interval_hours"].iloc[:-1]})
    unexpected = [h for h in observed if h not in allowed]
    if unexpected:
        problems.append(
            f"unexpected interval lengths (hours): {unexpected}; allowed "
            f"{sorted(allowed)} -- this usually means intervals are missing"
        )

    # Completeness against the elapsed UTC span. This is the DST-proof check:
    # it is correct for 23-, 24- and 25-hour local days without knowing which.
    covered = float(df["interval_hours"].sum())
    span = (utc_end - utc_start).total_seconds() / 3600.0
    if abs(covered - span) > 1e-6:
        problems.append(
            f"coverage mismatch: intervals span {covered:g} h but the requested "
            f"window is {span:g} h ({span - covered:+g} h missing)"
        )

    if problems:
        message = "; ".join(problems)
        if config.strict:
            raise PriceDataError(message)
        LOGGER.warning("price data issues (strict=false): %s", message)


def _build_frame(
    series: pd.Series, utc_start: pd.Timestamp, utc_end: pd.Timestamp, config: DataConfig
) -> pd.DataFrame:
    """Shape the raw ENTSO-E series into the module's contract."""
    series = series.tz_convert("UTC").sort_index()
    series = series[(series.index >= utc_start) & (series.index < utc_end)]
    series = series[~series.index.duplicated(keep="first")]

    df = pd.DataFrame({PRICE_COL: series.astype("float64")})
    df.index = pd.DatetimeIndex(df.index, name=INDEX_NAME)
    # Parquet does not preserve an inferred freq, so drop it here too and keep
    # cache hits identical to fresh fetches. It would be misleading regardless:
    # a frame spanning a clock change or the quarter-hourly switch has no single
    # step, and interval length is carried per row by `interval_hours` instead.
    df.index.freq = None

    local = df.index.tz_convert(config.timezone)
    df["local_time"] = local
    # Local calendar day the interval is delivered in, stored tz-naive so it
    # survives parquet cleanly. Grouping on this gives 23/24/25-hour days.
    df["delivery_date_local"] = local.normalize().tz_localize(None)
    df["interval_hours"] = _interval_hours(df.index)
    return df


def _log_day_lengths(df: pd.DataFrame) -> None:
    """Surface non-24-hour delivery days so clock changes are visible in logs."""
    hours_per_day = df.groupby("delivery_date_local")["interval_hours"].sum()
    for day, hours in hours_per_day[hours_per_day != 24.0].items():
        LOGGER.info("clock change: %s is a %g-hour delivery day", day.date(), hours)


def fetch_day_ahead_prices(
    start: DateLike,
    end: DateLike,
    *,
    config: Optional[DataConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    force_refresh: bool = False,
    use_cache: bool = True,
    client=None,
    **config_overrides,
) -> pd.DataFrame:
    """Fetch day-ahead prices for a range of local delivery dates.

    Parameters
    ----------
    start, end
        Local delivery dates, **inclusive on both ends**. Anything
        ``pd.Timestamp`` accepts works; a value carrying a time of day is taken
        as the delivery day it falls on.
    config, config_path, **config_overrides
        Settings source. Defaults to ``config/data.yaml``; pass e.g.
        ``zone="NL"`` to override a single field.
    force_refresh
        Re-query ENTSO-E and overwrite the cached parquet.
    use_cache
        Set false to bypass the cache entirely (neither read nor write).
    client
        Pre-built ``EntsoePandasClient``; mainly a test seam.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``timestamp_utc`` (tz-aware UTC, unique, increasing), with
        columns ``price_eur_mwh``, ``local_time`` (tz-aware zone local),
        ``delivery_date_local`` (naive local midnight) and ``interval_hours``.
        Each row describes the interval *starting* at its index value.
    """
    cfg = config or load_data_config(config_path, **config_overrides)
    utc_start, utc_end = delivery_window_utc(start, end, cfg.timezone)
    path = cache_path(start, end, cfg)

    if use_cache and path.exists() and not force_refresh:
        LOGGER.info("cache hit: %s", path)
        df = _read_cache(path)
        _validate(df, utc_start, utc_end, cfg)
        return df

    series = _download(utc_start, utc_end, cfg, client=client)
    df = _build_frame(series, utc_start, utc_end, cfg)
    _validate(df, utc_start, utc_end, cfg)
    _log_day_lengths(df)

    if use_cache:
        metadata = {
            "zone": cfg.zone,
            "timezone": cfg.timezone,
            "start_date_local": str(_as_delivery_date(start).date()),
            "end_date_local": str(_as_delivery_date(end).date()),
            "utc_start": utc_start.isoformat(),
            "utc_end_exclusive": utc_end.isoformat(),
            "n_intervals": int(len(df)),
            "interval_hours_present": sorted({float(h) for h in df["interval_hours"]}),
            "span_hours": float(df["interval_hours"].sum()),
            "price_column": PRICE_COL,
            "source": "ENTSO-E Transparency Platform, day-ahead prices (A44, SDAC)",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_cache(df, path, metadata)
        LOGGER.info("cached %d intervals -> %s", len(df), path)

    return df


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache ENTSO-E day-ahead prices.")
    parser.add_argument("--start", required=True, help="first local delivery date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last local delivery date, inclusive")
    parser.add_argument("--zone", default=None, help="override the configured bidding zone")
    parser.add_argument("--config", default=None, help="path to the ingestion YAML")
    parser.add_argument("--force-refresh", action="store_true", help="ignore any cached parquet")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {"zone": args.zone} if args.zone else {}
    df = fetch_day_ahead_prices(
        args.start,
        args.end,
        config_path=args.config,
        force_refresh=args.force_refresh,
        **overrides,
    )

    hours_per_day = df.groupby("delivery_date_local")["interval_hours"].sum()
    print(f"{len(df)} intervals, {df.index[0]} .. {df.index[-1]} (UTC)")
    print(
        f"delivery days: {len(hours_per_day)}  "
        f"non-24h days: {int((hours_per_day != 24.0).sum())}"
    )
    print(df[PRICE_COL].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
