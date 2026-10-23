import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from SpecMamba_Config import make_config
from SpecMamba_Data import build_dataloaders
from SpecMamba_Implementation import SpecSSM
from SpecMamba_Train import evaluate


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "weights" / "checkpoints"
PREFIXES = (
    ("model.revin.affine_weight", "revin.weight"),
    ("model.revin.affine_bias", "revin.bias"),
    ("model.trend_projector.", "trend."),
    ("model.patch_embedding.position_embedding", "patch.position"),
    ("model.patch_embedding.value_embedding.", "patch.projection."),
    ("model.mamba_layers.", "mamba_layers."),
    ("model.spec_layer.w_real", "spectral.real"),
    ("model.spec_layer.w_imag", "spectral.imag"),
    ("model.seasonal_head.", "head."),
    ("model.spec_gate", "alpha"),
)


def manifest_entry(dataset, horizon):
    with (CHECKPOINT_DIR / "MANIFEST.csv").open(newline="", encoding="utf-8-sig") as handle:
        matches = [row for row in csv.DictReader(handle) if row["Dataset"] == dataset and int(row["Horizon"]) == horizon]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint for {dataset}, T={horizon}; found {len(matches)}")
    return matches[0]


def remap_state(state):
    result = {}
    for key, value in state.items():
        for old, new in PREFIXES:
            if key == old or key.startswith(old) and old.endswith("."):
                renamed = new + key[len(old):]
                if renamed in result:
                    raise ValueError(f"Duplicate mapped parameter: {renamed}")
                result[renamed] = value
                break
        else:
            raise ValueError(f"Unsupported checkpoint parameter: {key}")
    return result


def checkpoint_config(dataset, horizon, state, data_dir, num_workers):
    projection = state["model.patch_embedding.value_embedding.weight"]
    width, patch_len = projection.shape
    head = state["model.seasonal_head.weight"]
    if head.shape[0] != horizon or head.shape[1] % width:
        raise ValueError("Checkpoint head shape does not match the selected horizon")
    patches = head.shape[1] // width
    seq_len = state["model.trend_projector.weight"].shape[1]
    strides = [stride for stride in range(1, seq_len + 1) if (seq_len + stride - patch_len) // stride + 1 == patches]
    if len(strides) != 1:
        raise ValueError(f"Cannot determine a unique stride from {patches} patches")
    if state["model.patch_embedding.position_embedding"].shape != (1, patches + 5, width):
        raise ValueError("Checkpoint position embedding shape is inconsistent")
    if state["model.spec_layer.w_real"].shape != (patches // 2 + 1, width):
        raise ValueError("Checkpoint frequency filter shape is inconsistent")
    layers = sorted({int(key.split(".")[2]) for key in state if key.startswith("model.mamba_layers.")})
    if layers != list(range(len(layers))):
        raise ValueError("Checkpoint Mamba layers are not consecutive")
    state_size = state["model.mamba_layers.0.1.A_log"].shape[1]
    return make_config(
        dataset,
        horizon,
        data_dir=data_dir,
        seq_len=seq_len,
        d_model=width,
        d_state=state_size,
        patch_len=patch_len,
        stride=strides[0],
        e_layers=len(layers),
        num_workers=num_workers,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    row = manifest_entry(args.dataset, args.horizon)
    path = CHECKPOINT_DIR / row["CopiedFile"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["SHA256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    config = checkpoint_config(args.dataset, args.horizon, state, args.data_dir, args.num_workers)
    loaders, _ = build_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpecSSM(config).to(device)
    model.load_state_dict(remap_state(state), strict=True)
    metrics = evaluate(model, loaders["test"], device)
    result = {"Dataset": args.dataset, "T": args.horizon, "Checkpoint": row["CopiedFile"], "Config": {key: config[key] for key in ("seq_len", "d_model", "d_state", "patch_len", "stride", "e_layers")}, **metrics}
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
