import argparse

import pandas as pd
from scipy.stats import ttest_ind


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
HORIZONS = (96, 192, 336, 720)


def run_level_means(path, model_name):
    frame = pd.read_csv(path)
    required = {"Model", "Dataset", "Seed", "T", "MSE", "MAE"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame.columns))}")
    frame = frame[frame["Model"] == model_name]
    frame = frame[frame["Dataset"].isin(DATASETS) & frame["T"].isin(HORIZONS)]
    if frame.duplicated(["Dataset", "Seed", "T"]).any():
        raise ValueError(f"Duplicate dataset, seed, horizon rows for {model_name}")
    counts = frame.groupby(["Dataset", "Seed"])["T"].nunique()
    if len(counts) != 25 or not counts.eq(4).all():
        raise ValueError(f"Expected five complete four-horizon runs per dataset for {model_name}")
    if frame.groupby("Dataset")["Seed"].nunique().ne(5).any():
        raise ValueError(f"Expected five seeds per dataset for {model_name}")
    return frame.groupby(["Dataset", "Seed"], as_index=False)[["MSE", "MAE"]].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="repeat_statistics.csv")
    parser.add_argument("--ours-name", default="Spec-SSM")
    parser.add_argument("--baseline-name", default="S-Mamba")
    args = parser.parse_args()
    ours = run_level_means(args.input, args.ours_name)
    baseline = run_level_means(args.input, args.baseline_name)
    rows = []
    for dataset in DATASETS:
        for metric in ("MSE", "MAE"):
            a = ours.loc[ours["Dataset"] == dataset, metric].to_numpy()
            b = baseline.loc[baseline["Dataset"] == dataset, metric].to_numpy()
            statistic, p_value = ttest_ind(a, b, equal_var=False)
            rows.append({"Dataset": dataset, "Metric": metric, "OursMean": a.mean(), "OursStd": a.std(ddof=1), "BaselineMean": b.mean(), "BaselineStd": b.std(ddof=1), "WelchT": statistic, "PValue": p_value})
    ranked = sorted(range(len(rows)), key=lambda index: rows[index]["PValue"])
    running = 0.0
    for rank, index in enumerate(ranked):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["PValue"]))
        rows[index]["HolmPValue"] = running
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
