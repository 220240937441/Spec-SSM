import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from SpecMamba_Data import build_dataloaders
from SpecMamba_Implementation import SpecSSM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    absolute_error = 0.0
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device))
            difference = prediction.float() - y.to(device)
            absolute_error += difference.abs().sum().item()
            squared_error += difference.square().sum().item()
            count += difference.numel()
    if count == 0:
        raise ValueError("Evaluation loader is empty")
    return {"MSE": squared_error / count, "MAE": absolute_error / count, "RMSE": (squared_error / count) ** 0.5}


def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(config, tag="full", save_checkpoint=True):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, borders = build_dataloaders(config)
    model = SpecSSM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"], eta_min=1e-6)
    criterion = nn.MSELoss()
    run_name = f"{config['dataset']}_L{config['seq_len']}_T{config['horizon']}_{tag}_seed{config['seed']}"
    run_dir = Path(config["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if model.spectral is not None:
        np.savez_compressed(run_dir / "initial_filter.npz", real=model.spectral.real.detach().cpu().numpy(), imag=model.spectral.imag.detach().cpu().numpy())
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "borders": borders}, handle, indent=2)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        training_loss = 0.0
        batches = 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            training_loss += loss.item()
            batches += 1
        if batches == 0:
            raise ValueError("Training loader is empty")
        validation = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": training_loss / batches, "val_MSE": validation["MSE"], "val_MAE": validation["MAE"]})
        if validation["MSE"] < best_loss:
            best_loss = validation["MSE"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= config["patience"]:
            break
    model.load_state_dict(best_state)
    metrics = evaluate(model, loaders["test"], device)
    gate = float(model.alpha.detach().cpu()) if model.alpha is not None else None
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    if save_checkpoint:
        torch.save({"model_state_dict": best_state, "config": config, "borders": borders, "best_epoch": best_epoch}, run_dir / "best.pt")
    summary = {"Model": "Spec-SSM", "Dataset": config["dataset"], "L": config["seq_len"], "T": config["horizon"], "Variant": tag, "Seed": config["seed"], "MSE": metrics["MSE"], "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "BestEpoch": best_epoch, "Alpha": gate}
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    append_result(Path(config["output_dir"]) / "results.csv", summary)
    return summary


def evaluate_checkpoint(checkpoint_path, data_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config["data_dir"] = data_dir
    loaders, borders = build_dataloaders(config)
    if borders != checkpoint["borders"]:
        raise ValueError("Dataset split differs from the checkpoint split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpecSSM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate(model, loaders["test"], device)
