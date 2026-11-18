# EP67 — State Space Models and Mamba

A small, educational PyTorch implementation exploring **state space models (SSMs)** and the ideas behind **Mamba**. It compares three sequence-mixing approaches on a synthetic memory-retrieval task:

1. **Causal self-attention** — a Transformer-style baseline.
2. **Fixed SSM** — a state-space recurrence whose step parameters do not depend on the current input.
3. **Selective SSM (Mamba-style)** — the step size and input/output state projections depend on the input.

> **Scope and accuracy note:** This repository is a teaching implementation, not the official `mamba-ssm` package. The SSM recurrence is written as a simple Python loop in PyTorch; it does not use Mamba's fused selective-scan CUDA kernels. Treat the timings as measurements of this implementation on your machine, not as a benchmark of production Mamba or a claim that this code is faster than optimized attention.

## What you will learn

- How a state can be updated recurrently:
  \[
  h_t = \bar A_t h_{t-1} + \bar B_t u_t,\qquad y_t = C_t h_t
  \]
- Why a **fixed** linear time-invariant SSM can also be expressed as a causal convolution.
- What changes when the state-space parameters become input-dependent (selectivity).
- How the sequence-length cost and memory accounting differ conceptually between attention and an SSM.
- How to train on short sequences and evaluate on longer ones.

## Repository contents

```text
.
├── ep67_mamba.py          # Complete experiment and model implementation
├── requirements.txt       # Python dependencies
├── README.md              # This guide
├── .gitignore
└── tests/
    └── test_smoke.py      # Fast checks for the core implementation
```

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- Matplotlib
- CPU is supported; a CUDA GPU is not required.

The experiment is intentionally small, but a full run can still take several minutes on a CPU. The script limits PyTorch to at most four CPU threads.

## Setup

### Windows (Command Prompt)

```bat
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-FOLDER>

py -3.10 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.10 is not installed, use another supported Python version, for example:

```bat
py -3.13 -m venv .venv
```

### macOS / Linux

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-FOLDER>

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the experiment

From the repository root:

```bash
python ep67_mamba.py
```

The script runs these experiments:

1. Checks recurrence-versus-convolution equivalence for a fixed SSM.
2. Benchmarks one attention layer and one SSM layer at several sequence lengths.
3. Trains attention, fixed SSM and selective SSM models on the synthetic memory task.
4. Evaluates the trained models on sequence lengths `32, 64, 128, 256, 512`.
5. Saves the results and dashboard.

Expected output files:

```text
mamba/
└── dashboard.png

ep67_results.json
```

The `mamba/` directory and results JSON are created next to `ep67_mamba.py`. The exact timings and accuracy values depend on hardware, software versions, and random seeds.

## Understanding the synthetic memory task

Each input sequence contains:

- Noise tokens represented by `0`.
- `M=4` data tokens, each randomly placed among the first `L` positions.
- Four query tokens appended at the end.

The target is the four data tokens in their original left-to-right order. The model predicts the answer tokens at the end of the sequence.

The models train only at `TRAIN_L=32`. They are then evaluated at lengths up to `512` (16 times the training sequence length). This is a controlled toy task, not a comprehensive test of language understanding or general long-context ability.

## The three models

| Model name in output | Description |
|---|---|
| `attention` | Causal multi-head self-attention with sinusoidal position encoding |
| `ssm_fixed` | State-space block with fixed step size and fixed B/C parameters |
| `mamba` | Selective state-space block where step size and B/C are generated from the input |

All three use the same general embedding, residual-block and prediction-head structure so the experiment can compare them on the same task.

## Recurrence and convolution

For a fixed SSM, the recurrence can be expanded into a convolution with kernel:

\[
K_k = C A^k B
\]

The script computes the output both ways and reports the maximum absolute difference. A small floating-point difference is expected.

The selective SSM changes its parameters based on the input. That input dependence is what makes the simple fixed convolution equivalence no longer directly applicable in the same way.

## Reading the benchmark carefully

The script reports:

- Seconds per forward pass for each tested sequence length.
- Log-log fitted slopes for the measured timings.
- A theoretical attention score count (`2 * L * L` in this script).
- A simple attention cache count and the fixed-size SSM state count.

These counts are illustrative accounting values, not a complete measurement of actual GPU/CPU memory use. The benchmark uses batch size 1, a small model width, and a simple Python-loop SSM. Python-loop overhead can dominate at these toy sizes. Do not compare these timings directly with published Mamba or Transformer benchmarks.

## Configuration

The experiment settings are near the top of `ep67_mamba.py`:

| Setting | Default | Meaning |
|---|---:|---|
| `SEEDS` | 2 | Independent training runs per model |
| `D_MODEL` | 32 | Model width |
| `N_STATE` | 8 | State size |
| `LAYERS` | 2 | Number of model blocks |
| `TRAIN_L` | 32 | Training sequence length |
| `STEPS` | 500 | Optimizer steps per run |
| `BATCH` | 64 | Training batch size |
| `TEST_LENS` | 32–512 | Evaluation sequence lengths |
| `BENCH_LENS` | 128–4096 | Benchmark sequence lengths |

For a quicker development run, temporarily lower `STEPS`, `SEEDS`, or the benchmark lengths. Restore the defaults for the episode's full experiment. Changing settings means your results will no longer match the default configuration.

## Run the quick checks

The smoke tests check that:

- The fixed-SSM recurrence and convolution agree within numerical tolerance.
- Each model returns logits with the expected shape.
- The outputs are finite.

```bash
python -m unittest discover -s tests -v
```

These checks do not run the full training experiment.

## Troubleshooting

**`ModuleNotFoundError` for torch, numpy or matplotlib**

Activate your virtual environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

**The full experiment takes too long**

The default run performs six training runs (three model types × two seeds), in addition to the benchmarks. Reduce `STEPS` and `SEEDS` while testing, then restore the defaults for a full run.

**The dashboard is not created**

Check that the script completed without an exception and that Matplotlib installed successfully. The expected file is `mamba/dashboard.png`.

**Results differ from another computer**

This is expected. CPU model, thread scheduling, PyTorch version and floating-point operations affect timing and can affect small accuracy differences. Compare the configuration and software versions recorded in `ep67_results.json`.

## Limitations

- Educational toy-scale models and synthetic data only.
- The SSM is implemented with a Python time-step loop, not a parallel scan or fused GPU kernel.
- The implementation is not a drop-in replacement for the official Mamba architecture or package.
- Two seeds provide only a small sample; reported standard deviations are not strong statistical guarantees.
- The memory task tests one narrow retrieval behavior and should not be interpreted as a broad measure of model quality.
- The benchmark measures this code path on the current machine; it is not a hardware-independent complexity proof.

## Suggested learning path

1. Run the smoke tests.
2. Read `equivalence()` and trace the recurrence and convolution calculations.
3. Read `SSM.forward()` and identify which values are fixed versus input-dependent.
4. Read `Attn.forward()` and compare its causal mask with the SSM update.
5. Run the full experiment and inspect `mamba/dashboard.png`.
6. Change one configuration value at a time and rerun to see how the experiment changes.

## Citation / attribution

This repository is an educational implementation based on the supplied EP67 episode challenge code. If you publish or adapt it, retain appropriate attribution to the original episode/author and add the source link here:

- Episode/video: **https://www.youtube.com/watch?v=_wN5HX_BbQg**


No license is declared by this template. Before making the repository public, choose a license only if you have the right to license all included material, and retain any required attribution from the original source.
