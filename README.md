# Dual-Band Multi-Target Sensing Network

This repository contains an inference-first release of a neural network for
joint target enumeration and range, velocity, and angle estimation from
heterogeneous high- and low-frequency ISAC sensing tensors.

The complete inference path contains three independently trained modules:

1. **CountNet** estimates the number of targets (`K=0...5`).
2. **ParamNet** predicts coarse range, velocity, and angle bins.
3. **PatchRefiner** extracts local `256 x 256 x 256` DenseFFT features around
   selected coarse estimates and returns continuous physical parameters.

Only CountNet's predicted top-`K` candidates enter the downstream modules.

## Repository contents

```text
models/       Model definitions and the assembled inference pipeline
training/     CountNet, ParamNet, and Refiner training programs
evaluation/   Count/runtime and K=2 parameter-CDF evaluation
data_tools/   Streaming Sionna NPZ-to-HDF5 conversion
scripts/      PowerShell command wrappers
weights/      Checkpoint manifest and download instructions
tests/        Lightweight release and interface tests
```

Converted caches, generated metrics, and checkpoint binaries are not tracked
by Git. The original datasets are hosted separately on Hugging Face.

## Installation

Python 3.8 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
pip install -r requirements.txt
```

## Pretrained weights

Download the three GitHub Release assets described in
[`weights/manifest.json`](weights/manifest.json) and place them directly under
`weights/`. See [`weights/README.md`](weights/README.md) for filenames and
checksum verification.

## Dataset

The Sionna-generated training and test datasets used by this project are
available from the
[DMSNet dataset repository on Hugging Face](https://huggingface.co/datasets/haotianbupt/DMSNet_dataset/tree/main).
Download the required `.npz` files from that repository and place them under
`data/DMSNet_dataset/`. The training set is used by the module-wise training programs, while
`beiyou_test_10000_clean.npz` and `beiyou_test_2000_clean_K2.npz` are used for
target-count/runtime evaluation and parameter-error evaluation, respectively.

Large files can also be obtained by cloning the dataset repository with Git
LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/haotianbupt/DMSNet_dataset data/DMSNet_dataset
```

## Prepare a dataset

```bash
python data_tools/convert_sionna_npz_to_h5.py \
  --npz data/DMSNet_dataset/beiyou_test_10000_clean.npz \
  --out data/beiyou_test_10000_clean_joint.h5
```

The converter applies the same common high/low-band scaling used for training.
See [`data_tools/DATA_FORMAT.md`](data_tools/DATA_FORMAT.md) for the required
tensor layout and label convention.

## Evaluation

Target-count metrics and complete CountNet-ParamNet-Refiner runtime:

```bash
python evaluation/evaluate_count_runtime.py \
  --test-h5 data/beiyou_test_10000_clean_joint.h5 \
  --snrs -10 -5 0 5 10 15 20
```

Matched-target parameter errors and P50/P90/P95 on the fixed `K=2` set:

```bash
python evaluation/evaluate_parameter_cdf.py \
  --test-h5 data/beiyou_test_2000_clean_K2_joint.h5 \
  --snrs 0
```

Runtime measures the complete neural forward path, including the Refiner's
internal DenseFFT processing, and excludes HDF5 loading, AWGN generation, and
metric computation.

## Training

Core training programs are retained for reproducibility. Their data and
checkpoint contract is documented in [`training/README.md`](training/README.md).
The public inference package does not require retraining.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall models training evaluation data_tools
```

## License

This project is released under the [MIT License](LICENSE).
