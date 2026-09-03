# Data Format

The public NPZ files are available from the
[DMSNet dataset repository on Hugging Face](https://huggingface.co/datasets/haotianbupt/DMSNet_dataset/tree/main).

The release expects clean complex Sionna tensors stored in an uncompressed or
ZIP-compressed NumPy archive (`.npz`). Large arrays are streamed during HDF5
conversion, so the complete dataset is not loaded into RAM.

## Required NPZ fields

| Field | Shape | Description |
| --- | --- | --- |
| `X_h` | `[N, 64, 112, 32]` | Complex high-band sensing tensor |
| `X_l` | `[N, 64, 14, 4]` | Complex low-band sensing tensor |
| `K_list` | `[N]` | Number of targets, from 0 to 5 |
| `Y_rva` or `Y` | `[N, 5, >=3]` | Range (m), velocity (m/s), angle (rad) |

Optional fields are `target_mask`, `P_h_clean`, and `P_l_clean`. When no mask
is supplied, the first `K_list[i]` target rows are treated as valid. Missing
power arrays default to one.

## Conversion

```bash
python data_tools/convert_sionna_npz_to_h5.py \
  --npz /path/to/dataset.npz \
  --out data/dataset.h5
```

The converter scales both bands with one common factor so that the median
non-empty high-band scene power equals `--target-p-h-ref` (default: 1.0). This
preserves the relative high/low-band power relationship.

The HDF5 output stores real and imaginary arrays separately as `float32`, plus
physical labels, count labels, scene powers, conversion metadata, and the
physical-grid configuration.
