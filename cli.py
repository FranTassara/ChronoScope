#!/usr/bin/env python3
"""
ChronoScope headless/batch CLI
===============================

Minimal command-line interface to ChronoScope's analysis engine, for use in
scripted or automated pipelines without launching the Qt GUI.

`core.analysis_engine.AnalysisEngine` and `utils.data_loader.CircadianDataLoader`
have no Qt/GUI dependencies, so this script drives them directly: it loads a
CSV/Excel file, runs one analysis method across the requested variable x
condition combinations, and writes the results to a CSV file.

Scope: this covers the tabular-data methods reachable through
AnalysisEngine.run_analysis (CosinorPy, CircaCompare, the Classical Rhythm
Analysis methods, RhythmCount, and CRS-AI). It does not cover DAM/AWD
locomotor recordings or single-cell RNA-seq loading, which use dedicated
loaders in the GUI (ui/analysis_panel.py) not wired up here.

Examples:
    python cli.py examples/synthetic_independent_test_data.csv --method jtk
    python cli.py data.csv --method cosinor_ols --time-col ZT --condition-col genotype
    python cli.py data.csv --method consensus_ai --variable Gene1 Gene2 --output ai_results.csv
    python cli.py data.csv --method lomb_scargle --param period_min=20 --param period_max=28
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.data_loader import CircadianDataLoader
from core.analysis_engine import AnalysisEngine, AnalysisType


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input', help='CSV file with time-series data')
    parser.add_argument(
        '--method', required=True, choices=sorted(t.value for t in AnalysisType),
        help='Analysis method to run',
    )
    parser.add_argument('--time-col', default=None, help='Time column name (auto-detected if omitted)')
    parser.add_argument('--condition-col', default=None, help='Condition column name (auto-detected if omitted)')
    parser.add_argument(
        '--variable', nargs='*', default=None,
        help='Variable column(s) to analyze (default: all auto-detected numeric variables)',
    )
    parser.add_argument(
        '--condition', nargs='*', default=None,
        help='Condition value(s) to analyze (default: all auto-detected conditions)',
    )
    parser.add_argument(
        '--param', action='append', default=[], metavar='KEY=VALUE',
        help='Method-specific parameter, repeatable, e.g. --param period_min=20 --param period_max=28',
    )
    parser.add_argument('--output', default=None, help='Output CSV path (default: <input>_<method>_results.csv)')
    return parser.parse_args()


def parse_params(param_list):
    params = {}
    for item in param_list:
        if '=' not in item:
            raise ValueError(f"--param must be KEY=VALUE, got: {item!r}")
        key, value = item.split('=', 1)
        try:
            value = float(value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            pass
        params[key] = value
    return params


def main():
    args = parse_args()
    parameters = parse_params(args.param)
    analysis_type = AnalysisType(args.method)

    loader = CircadianDataLoader()
    loader.load_csv(args.input)

    time_col = args.time_col or loader.get_time_column()
    if time_col is None:
        sys.exit("Could not auto-detect a time column; pass --time-col explicitly.")

    data = loader.get_data()
    condition_col = args.condition_col or loader.get_condition_column()
    if condition_col is None:
        condition_col = '_no_condition'
        data = data.assign(_no_condition='all')
        conditions = ['all']
    else:
        conditions = args.condition or loader.get_conditions()

    variables = args.variable or loader.get_variable_columns()
    if not variables:
        sys.exit("No numeric variable columns detected; pass --variable explicitly.")

    engine = AnalysisEngine()
    available, message = engine.check_method_available(analysis_type)
    if not available:
        sys.exit(f"Method '{args.method}' is not available: {message}")

    rows = []
    total = len(variables) * len(conditions)
    done = 0
    for variable in variables:
        for condition in conditions:
            result = engine.run_analysis(
                data=data,
                variable=variable,
                condition=condition,
                analysis_type=analysis_type,
                time_col=time_col,
                condition_col=condition_col,
                parameters=parameters,
            )
            # Some methods (e.g. a JTK period sweep) return a list of results
            # (one per tested period) rather than a single AnalysisResult.
            result_list = result if isinstance(result, list) else [result]
            rows.extend(asdict(r) for r in result_list)
            done += 1
            ok = all(r.success for r in result_list)
            status = 'OK' if ok else f"FAILED ({result_list[0].message})"
            print(f"[{done}/{total}] {variable} x {condition}: {status} ({len(result_list)} row(s))")

    out_path = args.output or f"{Path(args.input).stem}_{args.method}_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} result rows to {out_path}")


if __name__ == '__main__':
    main()
