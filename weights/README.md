# Pretrained Weights

The pretrained checkpoints are intentionally excluded from Git history. Their
combined size is about 194 MB, and the ParamNet checkpoint exceeds GitHub's
normal 100 MB per-file limit.

Publish the three files listed in `manifest.json` as GitHub Release assets and
place downloaded files directly in this directory:

```text
weights/
  countnet_sionna_best.pt
  paramnet_sionna_best.pt
  refiner_sionna_stage_d_best.pt
```

Verify a downloaded file on PowerShell with:

```powershell
Get-FileHash -Algorithm SHA256 .\weights\countnet_sionna_best.pt
```

The expected sizes and SHA256 values are stored in `manifest.json`.
