# Frontier Comparison Pipeline (HydraGNN GFM vs. MACE vs. UMA)

This directory contains the OLCF **Frontier** batch pipeline that benchmarks the
pretrained **HydraGNN Predictive GFM 2026** ensemble against two other
foundation models — **MACE** and **UMA** (fairchem) — on five datasets
(QM9, MD17, Wiggle150, ABC3, MS25), under identical train/val/test splits,
reporting both **accuracy** (per-atom energy MAE) and **wall-clock timing**.

For every dataset the pipeline runs up to five stages:

1. **HydraGNN fine-tune** — fine-tunes the GFM ensemble (frozen, unfrozen, and
   recycled-head variants).
2. **MACE fine-tune** — fine-tunes MACE foundation checkpoints (+ LoRA variant).
3. **UMA fine-tune** — fine-tunes UMA (full, frozen-backbone, and LoRA variants).
4. **UMA zero-shot benchmark**.
5. **MACE zero-shot benchmark**.

All five dataset pipelines run in **parallel** (one Frontier node each) inside a
single batch job, with skip guards so completed steps are not repeated on resubmission.

## Files

| File | Purpose |
|---|---|
| `submit_all.sbatch` | **Primary script.** Runs all 5 dataset pipelines in parallel (5 nodes, debug QOS). |
| `env_frontier.sh` | Shared environment setup (ROCm modules, venv activation, OLCF proxy, cache dirs). Sourced by `submit_all.sbatch`. |
| `build_mace_venv.sh` | Builds the dedicated MACE virtual environment (e3nn 0.4.4) on top of the HydraGNN venv. Run once per installation. |
| `prefetch_models.sh` | Downloads MACE/UMA foundation-model weights **on a login node** (compute nodes have no internet). Run once before submitting. |
| `submit_oqmd.sbatch` | Standalone job for the OQMD dataset (not included in `submit_all.sbatch`). |

## Prerequisites

1. **Base HydraGNN venv** and **MACE venv** built on Frontier (ROCm 7.2):
   ```bash
   bash frontier/build_mace_venv.sh
   ```
2. **Pretrained GFM ensemble** placed in `pretrained_model_ensemble/` — each
   member in its own subdirectory with a checkpoint and `config.json`.
3. **Foundation-model weights prefetched** on a login node:
   ```bash
   bash frontier/prefetch_models.sh
   ```
4. **All five datasets built** (see "Dataset acquisition" below).

## Reproducing the full benchmark

From the **repo root** on a Frontier login node:

```bash
export REPO_ROOT=$PWD
export FRONTIER_ROCM=7.2.0
export HYDRAGNN_VENV=$PWD/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
export MACE_VENV=$PWD/HydraGNN-Installation-Frontier-ROCm72/mace_venv_rocm72
sbatch --qos=debug frontier/submit_all.sbatch
```

The job requests 5 nodes and runs under the `debug` QOS (2-hour wall limit).
Each dataset pipeline writes its own log to `logs/frontier/all-<JOBID>-<dataset>.log`.

Results accumulate in `examples/<dataset>/benchmark_results/` and are
automatically skipped on resubmission if the output JSON already exists.
To force a full re-run, delete the relevant `benchmark_results/*.json` files.

> **Note on large-dataset UMA fine-tuning:** ABC3 (3185 train structures) and
> MS25 (5 VASP systems, 5000–8000 structures each) require ~55 min/epoch for
> UMA fine-tuning — infeasible within the 2-hour debug window. These variants
> are permanently skipped in `submit_all.sbatch`.  MACE fine-tuning on MS25 is
> similarly skipped for the same reason.

## UMA fine-tuning: catastrophic forgetting

Full-model UMA fine-tuning consistently degraded performance relative to
zero-shot (e.g. QM9: 9.9 → 720 meV/atom after 10 epochs). The pipeline runs
three UMA fine-tune variants for comparison:

| Variant | Flag | Behaviour |
|---|---|---|
| `full` | *(none)* | All weights updated — worst degradation |
| `frozen` | `--freeze-backbone` | Only the head is trained |
| `lora` | `--lora` | LoRA adapters on attention layers |

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
