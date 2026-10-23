import argparse
import json
from pathlib import Path

import numpy as np
import torch

from SpecMamba_Data import build_dataloaders
from SpecMamba_Implementation import SpecSSM


def load_model(checkpoint_path, data_dir):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(saved["config"])
    config["data_dir"] = data_dir
    loaders, borders = build_dataloaders(config)
    if borders != saved["borders"]:
        raise ValueError("Dataset split differs from the saved checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpecSSM(config).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, loaders["test"], config, device


def interpolate_missing(x, rate, generator):
    values = x.numpy().copy()
    mask = torch.rand(x.shape, generator=generator).numpy() < rate
    positions = np.arange(x.shape[1])
    for sample in range(x.shape[0]):
        for channel in range(x.shape[2]):
            observed = ~mask[sample, :, channel]
            if observed.any():
                values[sample, :, channel] = np.interp(positions, positions[observed], values[sample, observed, channel])
            else:
                values[sample, :, channel] = 0
    return torch.from_numpy(values)


def robustness(model, loader, device, seed):
    rows = []
    settings = [("noise", value) for value in (0, 0.05, 0.10, 0.20, 0.30)] + [("missing", value) for value in (0, 0.10, 0.20, 0.30, 0.40)]
    with torch.no_grad():
        for kind, value in settings:
            generator = torch.Generator().manual_seed(seed)
            squared_error = 0.0
            count = 0
            for x, y in loader:
                if kind == "noise" and value:
                    x = x + torch.randn(x.shape, generator=generator) * value
                if kind == "missing" and value:
                    x = interpolate_missing(x, value, generator)
                difference = model(x.to(device)).float() - y.to(device)
                squared_error += difference.square().sum().item()
                count += difference.numel()
            rows.append({"Perturbation": kind, "Level": value, "MSE": squared_error / count})
    return rows


def export_forecasts(model, loader, device, output, sample_index, channel_index):
    all_mse = []
    all_mae = []
    position = 0
    selected = None
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device)).float().cpu()
            error = prediction - y
            all_mse.append(error.square().mean(dim=2).numpy())
            all_mae.append(error.abs().mean(dim=2).numpy())
            if selected is None and position <= sample_index < position + x.size(0):
                offset = sample_index - position
                selected = (x[offset, :, channel_index].numpy(), y[offset, :, channel_index].numpy(), prediction[offset, :, channel_index].numpy())
            position += x.size(0)
    if selected is None:
        raise ValueError("Sample index is outside the test split")
    output.parent.mkdir(parents=True, exist_ok=True)
    values = {"mse": np.concatenate(all_mse), "mae": np.concatenate(all_mae), "history": selected[0], "target": selected[1], "prediction": selected[2]}
    if model.spectral is not None:
        real = model.spectral.real.detach().cpu().numpy()
        imag = model.spectral.imag.detach().cpu().numpy()
        values.update({"filter_real": real, "filter_imag": imag, "filter_magnitude": np.hypot(real, imag)})
    if model.alpha is not None:
        values["alpha"] = np.array(float(model.alpha.detach().cpu()))
    np.savez_compressed(output, **values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--mode", choices=("robustness", "export"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--channel-index", type=int, default=0)
    args = parser.parse_args()
    model, loader, config, device = load_model(args.checkpoint, args.data_dir)
    if args.mode == "robustness":
        if config["dataset"] not in ("ETTm1", "Weather") or config["seq_len"] != 96 or config["horizon"] != 192:
            raise ValueError("Robustness protocol requires ETTm1 or Weather, L=96, T=192")
        result = robustness(model, loader, device, args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        if args.channel_index < 0 or args.channel_index >= config["n_features"]:
            raise ValueError("Channel index is outside the dataset")
        export_forecasts(model, loader, device, Path(args.output), args.sample_index, args.channel_index)


if __name__ == "__main__":
    main()
