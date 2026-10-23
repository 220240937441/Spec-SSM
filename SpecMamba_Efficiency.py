import argparse
import json
import time
from pathlib import Path

import torch

from SpecMamba_Data import build_dataloaders
from SpecMamba_Implementation import SpecSSM


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def inference_measurement(model, sample, device, repetitions):
    model.eval()
    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            model(sample)
        synchronize(device)
    elapsed = time.perf_counter() - start
    return {"latency_ms": 1000 * elapsed / repetitions, "throughput_samples_per_s": repetitions * sample.size(0) / elapsed}


def run(checkpoint_path, data_dir, repetitions):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU measurement requires CUDA")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(saved["config"])
    config["data_dir"] = data_dir
    device = torch.device("cuda")
    loaders, borders = build_dataloaders(config)
    if borders != saved["borders"]:
        raise ValueError("Dataset split differs from the saved checkpoint")
    model = SpecSSM(config).to(device)
    model.load_state_dict(saved["model_state_dict"])
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    first_x, _ = next(iter(loaders["test"]))
    one = inference_measurement(model, first_x[:1], device, repetitions)
    full_batch = inference_measurement(model, first_x[: config["batch_size"]], device, repetitions)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    batches = 0
    for x, y in loaders["train"]:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        batches += 1
    synchronize(device)
    training_seconds = time.perf_counter() - start
    return {
        "dataset": config["dataset"],
        "look_back": config["seq_len"],
        "horizon": config["horizon"],
        "parameter_count": parameters,
        "training_seconds_per_epoch": training_seconds,
        "training_batches": batches,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "batch1_latency_ms": one["latency_ms"],
        "batch16_throughput_samples_per_s": full_batch["throughput_samples_per_s"],
        "split_borders": borders,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    result = run(args.checkpoint, args.data_dir, args.repetitions)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
