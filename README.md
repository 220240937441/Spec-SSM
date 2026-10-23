# Spec-SSM code package

This folder contains a source-level implementation of the Spec-SSM architecture and experiment entry points described in the revised manuscript. The model uses channel-independent patch embeddings, three residual Mamba blocks by default, a parallel learnable Fourier filter along patch tokens, a trainable scalar fusion coefficient, a separate linear prediction branch, and RevIN.

The repository contains the code needed to rerun the manuscript experiments and 16 existing training checkpoints in `weights/checkpoints/`. The author checked the checkpoint-to-experiment assignments and the submission package. The supplied checkpoints cover the dataset and horizon pairs listed in the manifest; the full 32-setting benchmark can be rerun from the source. Checkpoint file integrity and Python syntax were checked while preparing the package. A full GPU rerun was not performed as part of packaging.

## Files

| File | Purpose |
|---|---|
| `SpecMamba_Config.py` | Dataset registry and experiment settings |
| `SpecMamba_Implementation.py` | RevIN, patch embedding, frequency filter, Spec-SSM |
| `SpecMamba_Data.py` | CSV loading, exact split borders, train-only standardization, windows |
| `SpecMamba_Train.py` | AdamW training, cosine schedule, early stopping, checkpoint and metrics |
| `SpecMamba_Runner.py` | CLI for main and diagnostic experiments |
| `evaluate_existing_checkpoint.py` | Evaluate a supplied Lightning checkpoint using its matching model dimensions |
| `SpecMamba_Diagnostics.py` | Robustness evaluation, forecasts and trained filter export |
| `SpecMamba_Efficiency.py` | Checkpoint-based GPU timing and memory measurement |
| `analyze_repeats.py` | Five-run Welch tests and Holm correction from combined result CSV |
| `analyze_dm.py` | Paired forecast-origin Diebold–Mariano test from aligned loss arrays |
| `datasets/README.md` | Dataset paths and public sources |
| `DATASET_SHA256.tsv` | Checksums of the dataset files used while preparing this package |
| `weights/checkpoints/` | Existing Lightning checkpoints, experiment assignments and SHA-256 manifest |

## Environment

The original environment used Python 3.10, CUDA 11.8, PyTorch 2.1.1, `mamba-ssm` 2.2.2 and `causal-conv1d` 1.4.0 on a single NVIDIA RTX 3090. Install a CUDA-compatible PyTorch build first, then the remaining packages. Building Mamba can require a matching CUDA toolkit, compiler and `ninja`. Refer to the [official Mamba installation instructions](https://github.com/state-spaces/mamba#installation) if the wheel build fails. The two Linux wheel files in the original work directory are environment-specific and are intentionally not redistributed here.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The commands above target a Linux CUDA 11.8 environment. Check the installed Torch and CUDA versions before installing Mamba. The code requires `mamba_ssm`; it does not silently substitute a Transformer.

## Data and protocol

Place the eight CSV files under `datasets/` using the paths in [`datasets/README.md`](datasets/README.md). Data are not included in this repository. Each CSV must have a `date` column followed by the listed numeric channels. The loader validates feature counts and finite values.

ETTh splits use row borders 0, 8,640, 11,520 and 14,400. ETTm splits use 0, 34,560, 46,080 and 57,600. The other datasets use 70% training, the remaining middle segment for validation and the final 20% for testing, with integer truncation. Validation and test inputs may use the preceding 96 rows as context; prediction targets remain inside their own split. A `StandardScaler` is fitted to training rows only. MSE and MAE are calculated on the standardized targets, across all test values.

Default settings follow the manuscript: look-back 96, horizons 96/192/336/720, patch length 16, stride 8, three Mamba blocks, batch size 16, 30 maximum epochs, MSE loss, AdamW, cosine learning-rate schedule, and early stopping. The hidden/state dimensions are 32/16 for ETTh, ETTm, Weather and Exchange, and 64/32 for Electricity and Traffic. The fusion coefficient is initialized to 0.01 and remains trainable. The default seed is 2026; use `--seed` to select another seed. The paper does not record a full per-run seed list, so this default must not be presented as a recovered historical seed.

## Run

```bash
python SpecMamba_Runner.py --mode single --dataset ETTm1 --horizon 96 --data-dir datasets --output-dir outputs
python SpecMamba_Runner.py --mode benchmark --data-dir datasets --output-dir outputs
python SpecMamba_Runner.py --mode ablation --data-dir datasets --output-dir outputs
python SpecMamba_Runner.py --mode alpha --data-dir datasets --output-dir outputs
python SpecMamba_Runner.py --mode sensitivity --data-dir datasets --output-dir outputs
python SpecMamba_Runner.py --mode depth_bins --data-dir datasets --output-dir outputs
python SpecMamba_Runner.py --mode fusion --data-dir datasets --output-dir outputs
python SpecMamba_Runner.py --mode repeated --data-dir datasets --output-dir outputs
python SpecMamba_Runner.py --mode single --dataset Weather --horizon 192 --seq-len 336 --data-dir datasets --output-dir outputs
```

`benchmark` covers eight datasets and four horizons. `ablation` covers the full model and the three component variants on ETTm1, Weather, Traffic and Exchange. `alpha` retrains with initial values 0, 0.01, 0.1, 0.5 and 1.0. `sensitivity` varies hidden size, state size, patch size and stride on four datasets at horizon 96. `depth_bins` varies Mamba depth and retained low-frequency FFT bins on ETTm1 and Weather at all four horizons. `fusion` runs gated, fixed-add and concatenation-plus-linear fusion. `repeated` runs five sequential seeds on the five datasets described in the manuscript. `--seq-len` supports additional history lengths, including 336 and 720.

Each run writes `config.json`, `history.csv`, `metrics.json`, `initial_filter.npz` when the spectral branch is active, and by default `best.pt` under a uniquely named directory in `outputs/`. The best validation-MSE checkpoint is used for test evaluation. The aggregate `outputs/results.csv` is appended to after each run. Re-running the same command can append duplicate rows; remove or archive old output before a fresh comparison. To evaluate a saved checkpoint:

```bash
python SpecMamba_Runner.py --mode evaluate --checkpoint outputs/ETTm1_L96_T96_full_seed2026/best.pt --data-dir datasets
```

For the revised robustness experiment, evaluate trained ETTm1 and Weather checkpoints at `L=96,T=192`:

```bash
python SpecMamba_Diagnostics.py --mode robustness --checkpoint outputs/Weather_L96_T192_full_seed2026/best.pt --data-dir datasets --output outputs/weather_robustness.json
```

The diagnostic script applies Gaussian noise with standard deviations 0, 0.05, 0.10, 0.20 and 0.30, or randomly masks 0%, 10%, 20%, 30% and 40% of standardized test inputs, then interpolates each historical channel and leaves targets unchanged. The same procedure can be applied to the `no_spec` checkpoint. To export an aligned test loss matrix, a selected forecast, learned filter values and the final fusion coefficient:

```bash
python SpecMamba_Diagnostics.py --mode export --checkpoint outputs/Weather_L96_T192_full_seed2026/best.pt --data-dir datasets --output outputs/weather_export.npz --sample-index 0 --channel-index 0
```

`mse` and `mae` in the export have shape `[test windows, horizon]` and average over channels; `history`, `target` and `prediction` are one selected channel from one test window. The initial filter weights are saved by the training script. Select the same test window and channel when comparing external baselines.

For timing and memory measurements on a trained checkpoint:

```bash
python SpecMamba_Efficiency.py --checkpoint outputs/Weather_L96_T192_full_seed2026/best.pt --data-dir datasets --output outputs/weather_efficiency.json
```

This reports parameter count, one training epoch time, peak allocated GPU memory, batch-1 latency and batch-16 throughput. It continues one training epoch from the checkpoint in memory without overwriting it. Hardware, warmup, software versions and precision affect measured values. These measurements must be run on the intended GPU before comparing with the manuscript's efficiency table.

## Statistical comparison

The paper's five-run Welch/Holm analysis requires results from both Spec-SSM and the official S-Mamba implementation. Prepare one CSV with `Model,Dataset,Seed,T,MSE,MAE`, one row per dataset, seed and horizon, with model labels `Spec-SSM` and `S-Mamba`. Every model must have five complete four-horizon runs per dataset. Then run:

```bash
python analyze_repeats.py --input combined_runs.csv --output repeat_statistics.csv
```

The script first averages each seed's four horizons, then computes sample standard deviations, two-sided Welch tests and one Holm correction over ten dataset/metric comparisons. Baseline results and checkpoints are not generated by this package.

For a paired comparison with iTransformer, export the same `[test windows, horizon]` loss arrays from its official implementation, with the same data split, target scaling and forecast-origin order. Then run:

```bash
python analyze_dm.py --ours outputs/ettm1_export.npz --baseline outputs/itransformer_ettm1_export.npz --dataset ETTm1 --output outputs/ettm1_dm.json
```

This utility averages losses over each origin's horizon and channels, then uses a Bartlett-weighted long-run variance with lags through `T-1` and a two-sided normal reference distribution. Its output should be compared with the manuscript only after confirming that the original analysis used the same loss aggregation and variance convention.

## Baseline implementations

The manuscript describes baselines run with their official code and recommended configurations. This repository does not include substitute implementations of those methods. Useful official sources include [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [iTransformer](https://github.com/thuml/iTransformer), [PatchTST](https://github.com/yuqinie98/PatchTST), and the [S-Mamba paper repository](https://github.com/wzhwzhwzh0921/S-D-Mamba). Preserve the upstream commit, command line and resulting metrics when reproducing a baseline.

## Supplied checkpoints

The 16 supplied Lightning checkpoints are indexed by dataset and horizon in [`weights/checkpoints/MANIFEST.csv`](weights/checkpoints/MANIFEST.csv). Their model dimensions are read from the checkpoint tensors when evaluating because some supplied runs used different dimensions from the current manuscript defaults. Use the dedicated entry point for these checkpoints:

```bash
python evaluate_existing_checkpoint.py --dataset ETTh1 --horizon 96 --data-dir datasets --output outputs/etth1_t96_checkpoint.json
```

See [`weights/checkpoints/README.md`](weights/checkpoints/README.md) for the covered settings. These Lightning files are not inputs to `SpecMamba_Runner.py --mode evaluate`, which reads the new `best.pt` format. Once this folder is uploaded, cite the actual repository URL in the manuscript's Code Availability statement and the reviewer response.
