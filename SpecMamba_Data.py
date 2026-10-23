import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from SpecMamba_Config import DATASETS, data_path


def split_borders(config, total_length):
    frequency = DATASETS[config["dataset"]][2]
    if frequency == "hourly":
        train_end, val_end, test_end = 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
    elif frequency == "15min":
        train_end, val_end, test_end = 12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4
    else:
        train_end = int(total_length * 0.7)
        test_start = total_length - int(total_length * 0.2)
        val_end, test_end = test_start, total_length
    if test_end > total_length:
        raise ValueError("Dataset is shorter than the stated ETT split")
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, test_end)}


class WindowDataset(Dataset):
    def __init__(self, values, start, end, seq_len, horizon, use_history):
        self.values = values
        self.start = start - seq_len if use_history else start
        self.count = end - self.start - seq_len - horizon + 1
        self.seq_len = seq_len
        self.horizon = horizon
        if self.start < 0 or self.count <= 0:
            raise ValueError("Split has no valid forecasting windows")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        position = self.start + index
        return self.values[position : position + self.seq_len], self.values[position + self.seq_len : position + self.seq_len + self.horizon]


def build_dataloaders(config):
    path = data_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing: {path}")
    frame = pd.read_csv(path)
    if "date" in frame:
        frame = frame.drop(columns="date")
    if len(frame.columns) != config["n_features"]:
        raise ValueError(f"Expected {config['n_features']} features, found {len(frame.columns)} in {path}")
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Missing or nonfinite values found in {path}")
    borders = split_borders(config, len(values))
    scaler = StandardScaler().fit(values[: borders["train"][1]])
    values = torch.from_numpy(scaler.transform(values).astype(np.float32))
    loaders = {}
    for name, (start, end) in borders.items():
        dataset = WindowDataset(values, start, end, config["seq_len"], config["horizon"], name != "train")
        workers = config["num_workers"]
        loaders[name] = DataLoader(dataset, batch_size=config["batch_size"], shuffle=name == "train", drop_last=name == "train", num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    return loaders, borders
