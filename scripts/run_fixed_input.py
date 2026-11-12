#!/usr/bin/env python3
"""Run the fixed-input Monte Carlo experiment behind Table 1 and Figure 6.

Usage:
    python scripts/run_fixed_input.py               sample new fields
    python scripts/run_fixed_input.py --summarize   rebuild the records from saved errors

Results go to results/fixed_input/. Notebook 03 reads them to build Table 1 and
Figure 6. The same seed can give slightly different Monte Carlo values in a
different numerical environment, so run_info.json records the package versions.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigf import fixed_input  # noqa: E402
from bigf.paths import relative  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summarize", action="store_true",
        help="rebuild calibration_records.json from results/fixed_input/raw/",
    )
    args = parser.parse_args()

    config = fixed_input.FixedInputConfig()
    results_dir = fixed_input.DEFAULT_RESULTS_DIR

    if args.summarize:
        records = fixed_input.summarize(config, results_dir)
        path = results_dir / "calibration_records.json"
        path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"Rebuilt {len(records)} records from saved errors: {relative(path)}")
        return

    print(f"Fixed-input calibration: {config.n_scenarios} scenarios, "
          f"{config.n_fields:,} fields per scenario, seed {config.seed}.")
    start = time.perf_counter()
    records = fixed_input.run(
        config, results_dir,
        progress=lambda name, d, r: print(f"  {name:<20} D={d:<4} R={r:g}", flush=True),
    )
    print(f"Saved {len(records)} records to {relative(results_dir)} "
          f"in {time.perf_counter() - start:.1f} s.")


if __name__ == "__main__":
    main()
