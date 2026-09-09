# Reproducibility

## CPU verification
```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/dev_smoke.py
```

## Free-GPU development
Open `notebooks/Paper4_End_to_End_Free_GPU.ipynb` in Kaggle or Colab with a CUDA GPU enabled. The notebook installs local/open dependencies, records `results/environment.json`, runs tests before inference, downloads the official A-OKVQA annotation archive plus COCO validation images for DEV mode, builds a leakage-safe manifest, and runs B0/B1/B2-B3 development inference.

## Split discipline
- Ground-truth answers may exist in manifest evaluation fields but are removed by `VQASample.inference_view()` before retrieval/generation.
- Calibration and threshold fitting must use a calibration/validation split only.
- A frozen calibration policy is applied to held-out predictions using `scripts/apply_selective_policy.py`.
- The final test/evaluation split must never be used to tune thresholds, coefficients, prompts, top-k, or model selection.

## Core commands
```bash
python scripts/build_manifest.py --help
python scripts/run_baseline.py --help
python scripts/run_proposed.py --help
python scripts/fit_selective_policy.py --help
python scripts/apply_selective_policy.py --help
```

## Environment capture
`python scripts/inspect_environment.py` writes exact runtime information to `results/environment.json`.

## Paper-integrity rule
Synthetic smoke outputs and DEV-mode subset metrics are engineering checks only. They are not manuscript benchmark results.
