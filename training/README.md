# Training

The public training code mirrors the independently trained modules used by the
released inference pipeline:

1. `train_countnet.py` trains target enumeration from the clean Sionna NPZ.
2. `train_paramnet.py` trains coarse range/velocity/angle estimation using the
   converted HDF5 dataset and a trained CountNet checkpoint.
3. `train_refiner_stage_a.py` through `train_refiner_stage_d.py` implement the
   Refiner curriculum; only the final Stage-D checkpoint is used for inference.

## Paths

CountNet reads the source NPZ from the `SIONNA_TRAIN_NPZ` environment variable:

```powershell
$env:SIONNA_TRAIN_NPZ = "C:\path\to\train.npz"
python training/train_countnet.py
```

ParamNet and Refiner expect the converted training file at:

```text
data/beiyou_train_clean_50000_paramnet.h5
```

Place the selected CountNet checkpoint at
`weights/countnet_sionna_best.pt` before ParamNet training, and place the
selected ParamNet checkpoint at `weights/paramnet_sionna_best.pt` before
Refiner training. Training outputs are written under `outputs/`.

The published defaults reproduce the selected architecture and optimization
settings. Dataset locations and experiment outputs are release-relative; no
source-tree file outside this package is read or modified.
