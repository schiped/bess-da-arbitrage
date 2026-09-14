# BESS Day-Ahead Arbitrage

Stack: Python 3.11, Pyomo + HiGHS, pandas, entsoe-py
Data: ENTSO-E Transparency API, DE-LU day-ahead, 2023-2024

## Modelling rules
- Separate charge/discharge efficiency (0.92 each), never lumped
- Degradation as throughput cost in €/MWh cycled
- SoC in MWh, power in MW — be explicit in variable names
- Rolling horizon must respect 12:00 D-1 gate closure
- End-of-horizon SoC valuation required in objective

## Conventions
- Config in YAML, no hardcoded parameters
- Every optimisation run writes a results CSV + assumptions dump