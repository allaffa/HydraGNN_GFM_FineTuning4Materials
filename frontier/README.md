# Frontier Comparison Pipeline (HydraGNN GFM vs. MACE vs. UMA)

This directory contains the OLCF **Frontier** batch pipeline that benchmarks the
pretrained **HydraGNN Predictive GFM 2026** ensemble against two other
foundation models — **MACE** and **UMA** (fairchem) — on the same datasets,
under identical train/val/test splits, reporting both **accuracy** (per-atom
energy MAE) and **wall-clock timing**.

For every dataset the pipeline runs a five-stage job:

1. **HydraGNN benchmark** — trains/evaluates several fine-tuning strategies
   (`scratch`, `unfrozen`, `frozen`, and recycled-head variants) from the GFM
   ensemble.
2. **MACE fine-tune** — fine-tunes MACE foundation checkpoints on the split.
3. **UMA fine-tune** — fine-tunes the UMA model (see `--freeze-backbone` note
   below).
4. **UMA zero-shot benchmark**.
5. **MACE zero-shot benchmark** and assembly of the final comparison table.

## Files

| File | Purpose |
|---|---|
| `env_frontier.sh` | Shared environment setup (ROCm modules, venv activation, OLCF proxy, cache dirs). Sourced by every batch script. |
| `build_mace_venv.sh` | Builds the dedicated MACE virtual environment (e3nn 0.4.4) layered on the base HydraGNN venv. |
| `prefetch_models.sh` | Downloads MACE/UMA foundation-model weights **on a login node** (compute nodes have no internet). Run this once before submitting jobs. |
| `submit_<dataset>.sbatch` | One batch job per dataset: `md17`, `qm9`, `ms25`, `abc3`, `oqmd`, `wiggle150`. |

## Prerequisites

1. **Base HydraGNN venv** and **MACE venv** built on Frontier (ROCm 7.2).
2. **Pretrained GFM ensemble** downloaded (via Globus, see repo root README) into
   `pretrained_model_ensemble/` — each ensemble member in its own subdirectory
   with its checkpoint and `config.json`.
3. **Foundation-model weights prefetched** on a login node:
   ```bash
   bash frontier/prefetch_models.sh
   ```
4. **Dataset built** for the target example (see "Dataset acquisition" below).

## Submitting a job

From the **repo root**:

```bash
export REPO_ROOT=$PWD
export FRONTIER_ROCM=7.2.0
export HYDRAGNN_VENV=$PWD/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
export MACE_VENV=$PWD/HydraGNN-Installation-Frontier-ROCm72/mace_venv_rocm72
sbatch frontier/submit_qm9.sbatch      # or md17 / ms25 / abc3 / oqmd / wiggle150
```

All site-specific paths are overridable via environment variables, so the batch
scripts do not need to be edited per machine.

> **Note — walltime:** single-node (`-N 1`) `batch` jobs on Frontier are capped
> at **2 hours**. The submit scripts request `02:00:00`.

## UMA fine-tuning: `--freeze-backbone`

Full-model UMA fine-tuning on the small per-dataset splits caused **catastrophic
forgetting** (e.g. QM9 test MAE degraded from ~0.01 eV/atom zero-shot to
~0.72 eV/atom after 10 epochs). The pipeline therefore fine-tunes UMA with the
**backbone frozen**, mirroring HydraGNN's `frozen` strategy (its best performer).

- Most `submit_*.sbatch` scripts pass `--freeze-backbone` directly to
  `run_uma_finetune.py`.
- `submit_qm9.sbatch` exposes it via an environment variable:
  ```bash
  UMA_EXTRA_ARGS=--freeze-backbone sbatch --export=ALL frontier/submit_qm9.sbatch
  ```

## Dataset acquisition

Each dataset is fetched/preprocessed once (on a **login node** for the ones that
need internet), producing a pickle under `dataset/`:

| Dataset | Acquisition | Notes |
|---|---|---|
| `qm9` | `qm9_energy_preonly.py` | Downloaded automatically. |
| `md17` | `md17_mlip_preonly.py` | Downloaded automatically. |
| `ms25` | `ms25_preonly.py` | Produces per-atom pickles. |
| `abc3` | `abc3_getData_API.py` → `abc3_preonly.py --pickle` | Requires a **Materials Project API key** in `MP_API_KEY` (never stored in the repo; see `examples/abc3/api_keys.py`). |
| `oqmd` | `oqmd_getData.py` → `oqmd_preonly.py --pickle` | Uses the **public** OQMD REST API — no key needed. Long-running scan. |
| `wiggle150` | `wiggle150_preonly.py --pickle` | Publisher blocks automated download; place the XYZ manually at `dataset/wiggle150/raw/` (energies are read from the XYZ comment lines). |

### Login-node network notes (OLCF)

- Compute nodes have **no outbound internet**; download/prefetch on a login node.
- The OLCF squid proxy set by `env_frontier.sh` **breaks streaming downloads**.
  Before running a `*_getData*.py` step on a login node, unset the proxy and
  point at the system CA bundle:
  ```bash
  unset http_proxy https_proxy all_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  export REQUESTS_CA_BUNDLE=/etc/ssl/ca-bundle.pem \
         CURL_CA_BUNDLE=/etc/ssl/ca-bundle.pem \
         SSL_CERT_FILE=/etc/ssl/ca-bundle.pem
  ```

### `mp-api` / `emmet-core` compatibility

`mp-api 0.41.2` imports `BSPathType` from `emmet.core.electronic_structure`, but
newer `emmet-core` (≥0.85) moved it to `emmet.core.band_theory`, and `mp-api`
pins only `emmet-core>=0.78` (no upper bound). `abc3_getData_API.py` therefore
re-exposes `BSPathType` on the old module *before* importing `mp_api`, so the
Materials Project client imports cleanly without downgrading the shared venv.

## Outputs

Each job writes to `examples/<dataset>/benchmark_results/`:

- `benchmark_summary.json` — HydraGNN fine-tuning strategies.
- `mace_finetuned_summary.json`, `mace_benchmark_summary.json` — MACE.
- `uma_finetuned_summary.json`, `uma_benchmark_summary.json` — UMA.
- `val_histories.json` and a validation plot (`*_validation.png`).

The final MACE benchmark stage assembles the combined accuracy + timing table
across all three model families.
