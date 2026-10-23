import argparse
import json

from SpecMamba_Config import ABLATIONS, DATASETS, HORIZONS, make_config
from SpecMamba_Train import evaluate_checkpoint, run_experiment


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "benchmark", "ablation", "alpha", "sensitivity", "depth_bins", "fusion", "repeated", "evaluate"), default="single")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="ETTm1")
    parser.add_argument("--horizon", type=int, choices=HORIZONS, default=96)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser


def experiments(args):
    common = {"data_dir": args.data_dir, "output_dir": args.output_dir, "seed": args.seed, "seq_len": args.seq_len, "num_workers": args.num_workers}
    if args.mode == "single":
        yield make_config(args.dataset, args.horizon, **common), "full"
    elif args.mode == "benchmark":
        for dataset in DATASETS:
            for horizon in HORIZONS:
                yield make_config(dataset, horizon, **common), "full"
    elif args.mode == "ablation":
        for dataset in ("ETTm1", "Weather", "Traffic", "Exchange"):
            for horizon in HORIZONS:
                for tag, (trend, spec) in ABLATIONS.items():
                    yield make_config(dataset, horizon, use_trend=trend, use_spec=spec, **common), tag
    elif args.mode == "alpha":
        for dataset in ("ETTm1", "Weather"):
            for horizon in HORIZONS:
                for alpha in (0, 0.01, 0.1, 0.5, 1.0):
                    yield make_config(dataset, horizon, alpha_init=alpha, **common), f"alpha_{alpha:g}"
    elif args.mode == "sensitivity":
        for dataset in ("ETTm1", "Weather", "Traffic", "Electricity"):
            for key, values in {"d_model": (32, 64, 128, 256), "d_state": (8, 16, 32, 64), "patch_len": (8, 12, 16, 24), "stride": (4, 8, 12, 16)}.items():
                for value in values:
                    yield make_config(dataset, 96, **common, **{key: value}), f"{key}_{value}"
    elif args.mode == "depth_bins":
        for dataset in ("ETTm1", "Weather"):
            for horizon in HORIZONS:
                for depth in (1, 2, 3, 4):
                    yield make_config(dataset, horizon, e_layers=depth, **common), f"depth_{depth}"
                for bins in (2, 4, 6, 7):
                    yield make_config(dataset, horizon, filter_bins=bins, **common), f"bins_{bins}"
    elif args.mode == "fusion":
        for dataset in ("ETTm1", "Weather"):
            for horizon in HORIZONS:
                for fusion in ("fixed_add", "concat_mlp", "gated"):
                    yield make_config(dataset, horizon, fusion=fusion, **common), fusion
    elif args.mode == "repeated":
        for dataset in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"):
            for seed in range(args.seed, args.seed + 5):
                for horizon in HORIZONS:
                    yield make_config(dataset, horizon, **{**common, "seed": seed}), "full"


def main():
    args = argument_parser().parse_args()
    if args.mode == "evaluate":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for evaluate mode")
        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.data_dir), indent=2))
        return
    for config, tag in experiments(args):
        print(json.dumps(run_experiment(config, tag, not args.no_save_checkpoint), ensure_ascii=False))


if __name__ == "__main__":
    main()
