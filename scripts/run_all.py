#!/usr/bin/env python3
"""
Run the full pipeline end to end, in order, steps 1-10.

This is a convenience wrapper for a first full run or a from-scratch
reproduction check -- every step is independently resumable (each script
skips work it has already done), so re-running this after an interruption
picks up where it left off. For everyday development, prefer invoking the
individual numbered scripts directly, and see docs/pipeline.md for expected
runtime per step and how to parallelize across machines.

Steps 5 (grid search) and 8 (test evaluation) dominate total runtime; on a
single modern workstation, budget on the order of a day for the full
pipeline against EB-NeRD Large. NRMS reproducibility variants
(``--with-nrms-variants``) roughly triple NRMS-related runtime and are
optional -- they reproduce the paper's Section 5 reproducibility check, not
the primary results.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --with-mcf --with-nrms-variants
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def _run(args: list[str]) -> None:
    print(f"\n{'=' * 70}\n$ python {' '.join(args)}\n{'=' * 70}")
    subprocess.run([sys.executable, str(SCRIPTS_DIR / args[0]), *args[1:]], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-mcf", action="store_true", help="Also run the MCF baseline (step 7).")
    ap.add_argument("--with-nrms-variants", action="store_true",
                     help="Also train+evaluate the 3 extra NRMS instances for the reproducibility check.")
    args = ap.parse_args()

    _run(["01_prepare_data.py"])
    _run(["02_build_semantic_profiles.py"])
    _run(["03_score_base_models.py"])
    _run(["04_train_nrms.py"])  # primary seed0 run
    _run(["05_grid_search.py"])
    _run(["06_select_hyperparameters.py"])

    if args.with_mcf:
        _run(["07_mcf_baseline.py", "grid"])
        _run(["07_mcf_baseline.py", "select"])
        _run(["07_mcf_baseline.py", "eval"])

    _run(["08_evaluate_test.py"])

    if args.with_nrms_variants:
        for tag, params in [("seed1", ["--seed", "1"]),
                             ("seed2", ["--seed", "2"]),
                             ("large1M", ["--seed", "0", "--train-sample", "1000000"])]:
            _run(["04_train_nrms.py", "--tag", tag, *params])
            _run(["08_evaluate_test.py", "--models", "nrms", "--nrms-tag", tag])

    _run(["09_statistical_tests.py"])
    _run(["10_make_figures.py"])

    print("\nPipeline complete. See docs/pipeline.md for what each output file is and how it maps"
          " to the paper's tables and figures.")


if __name__ == "__main__":
    main()
