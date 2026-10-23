from pathlib import Path


DATASETS = {
    "ETTh1": ("ETT-small/ETTh1.csv", 7, "hourly"),
    "ETTh2": ("ETT-small/ETTh2.csv", 7, "hourly"),
    "ETTm1": ("ETT-small/ETTm1.csv", 7, "15min"),
    "ETTm2": ("ETT-small/ETTm2.csv", 7, "15min"),
    "Electricity": ("electricity/electricity.csv", 321, "ratio"),
    "Traffic": ("traffic/traffic.csv", 862, "ratio"),
    "Weather": ("weather/weather.csv", 21, "ratio"),
    "Exchange": ("exchange_rate/exchange_rate.csv", 8, "ratio"),
}

HORIZONS = (96, 192, 336, 720)
ABLATIONS = {
    "full": (True, True),
    "no_trend": (False, True),
    "no_spec": (True, False),
    "no_trend_spec": (False, False),
}


def make_config(dataset, horizon, data_dir="datasets", output_dir="outputs", **overrides):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    large = dataset in ("Electricity", "Traffic")
    config = {
        "dataset": dataset,
        "horizon": horizon,
        "data_dir": str(Path(data_dir)),
        "output_dir": str(Path(output_dir)),
        "seq_len": 96,
        "n_features": DATASETS[dataset][1],
        "d_model": 64 if large else 32,
        "d_state": 32 if large else 16,
        "d_conv": 4,
        "expand": 2,
        "e_layers": 3,
        "dropout": 0.2 if large else 0.1,
        "patch_len": 16,
        "stride": 8,
        "alpha_init": 0.01,
        "filter_bins": None,
        "fusion": "gated",
        "use_trend": True,
        "use_spec": True,
        "batch_size": 16,
        "max_epochs": 30,
        "patience": 5,
        "learning_rate": 0.001 if large else 0.0005,
        "num_workers": 4,
        "seed": 2026,
    }
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config.update(overrides)
    return config


def data_path(config):
    return Path(config["data_dir"]) / DATASETS[config["dataset"]][0]
