import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


def dm_test(ours, baseline):
    if ours.shape != baseline.shape or ours.ndim != 2:
        raise ValueError("Aligned loss arrays must have identical [windows, horizon] shape")
    windows, horizon = ours.shape
    if windows <= horizon:
        raise ValueError("Number of windows must exceed the forecast horizon")
    differential = baseline.mean(axis=1) - ours.mean(axis=1)
    centered = differential - differential.mean()
    long_run_variance = np.dot(centered, centered) / windows
    for lag in range(1, horizon):
        covariance = np.dot(centered[lag:], centered[:-lag]) / windows
        long_run_variance += 2 * (1 - lag / horizon) * covariance
    if long_run_variance <= 0:
        raise ValueError("Nonpositive long-run variance")
    statistic = differential.mean() / np.sqrt(long_run_variance / windows)
    return {"DM": float(statistic), "p_value": float(2 * norm.sf(abs(statistic))), "mean_baseline_minus_ours": float(differential.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = {"dataset": args.dataset}
    with np.load(args.ours) as ours, np.load(args.baseline) as baseline:
        for metric in ("mse", "mae"):
            results[metric] = dm_test(ours[metric], baseline[metric])
        results["horizon"] = int(ours["mse"].shape[1])
        results["windows"] = int(ours["mse"].shape[0])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
