#!/usr/bin/env python3
"""MD17 MACE state-of-the-art benchmark.

Evaluates three MACE foundation models (MACE-OFF23 medium, MACE-POLAR-1
polar-1-m, MACE-MH-1 omol head) on the MD17 uracil MLIP test split and
compares the results to the HydraGNN fine-tuning numbers produced by
``run_benchmark.py`` and optionally the UMA numbers from
``run_uma_benchmark.py``.

Dataset
-------
MD17 uracil MLIP pickle (``dataset/md17_mlip.pickle``), preprocessed by
``examples/md17/md17_mlip_preonly.py``.  Energies are in eV; forces in eV/Å.
Non-periodic single-molecule trajectories.

Energy reference treatment
--------------------------
Each MACE model uses a different DFT functional as reference, while the MD17
labels follow the ωB97X-D convention.  Both predicted and ground-truth energy
arrays are mean-centred before computing MAE so that the absolute reference
offset cancels — consistent with the UMA benchmark and the HydraGNN fine-tuning
benchmark that also trains on mean-shifted energies.  Forces are
reference-invariant and evaluated without any centering.

Usage
-----
    python examples/md17/run_mace_benchmark.py

    # Select a subset of models
    python examples/md17/run_mace_benchmark.py \\
        --models mace_off_medium mace_polar_m

    # Full comparison table (MACE + UMA + HydraGNN)
    python examples/md17/run_mace_benchmark.py \\
        --hydragnn-summary examples/md17/benchmark_results/benchmark_summary.json \\
        --uma-summary      examples/md17/benchmark_results/uma_benchmark_summary.json

Prerequisites
-------------
    pip install mace-torch>=0.3.16

References
----------
Kovacs et al., "MACE-OFF23", arXiv:2312.15211
Batatia et al., "MACE-MH-1",  arXiv:2510.25380
Batatia et al., "MACE-POLAR-1", arXiv:2602.19411
"""

from __future__ import annotations

import argparse
import json
import time
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from hydragnn.utils.distributed import setup_ddp
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

from utils.mace_calculator import MACE_MODELS, build_mace_calculator, pyg_data_to_ase_atoms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR = str(REPO_ROOT / "dataset" / "md17_mlip.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "md17" / "benchmark_results")
KCAL_PER_EV = 23.0609

_ALL_MODEL_IDS = list(MACE_MODELS.keys())

_VAR_CONFIG = {
    "type": ["graph"],
    "output_index": [0],
    "output_dim": [1],
    "output_names": ["graph_energy"],
    "graph_feature_names": ["energy"],
    "graph_feature_dims": [1],
    "node_feature_names": ["atomic_number"],
    "node_feature_dims": [1],
    "input_node_features": [0],
    "denormalize_output": False,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    split_label: str,
    model_id: str,
    device: str | None,
    verbose: bool = True,
) -> dict:
    """Evaluate a single MACE model on one MD17 split.

    Energies are mean-centred to cancel the DFT-reference offset.  Forces are
    evaluated without centering (reference-invariant).
    """
    dataset = SimplePickleDataset(
        basedir=DATASET_DIR, label=split_label, var_config=_VAR_CONFIG
    )
    if len(dataset) == 0:
        print(f"  [{split_label}] empty — skipping.")
        return {}

    calc = build_mace_calculator(model_id=model_id, device=device)
    needs_cs = MACE_MODELS[model_id]["needs_charge_spin"]

    e_preds, e_trues = [], []
    force_errors_flat = []
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        if needs_cs:
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
        atoms.calc = calc

        e_preds.append(float(atoms.get_potential_energy()))
        f_pred = atoms.get_forces()

        e_trues.append(float(data.energy.detach().cpu().squeeze()))
        f_true = data.forces.detach().cpu().numpy()

        force_errors_flat.append((f_pred - f_true).ravel())

        if verbose and (i + 1) % 100 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    _inference_sec = time.perf_counter() - _t0
    e_preds = np.asarray(e_preds)
    e_trues = np.asarray(e_trues)
    e_errors = (e_preds - e_preds.mean()) - (e_trues - e_trues.mean())
    force_flat = np.concatenate(force_errors_flat)

    return {
        "n_structures": len(e_errors),
        "inference_wall_sec": round(_inference_sec, 2),
        "energy_mae_eV": float(np.abs(e_errors).mean()),
        "energy_rmse_eV": float(np.sqrt((e_errors ** 2).mean())),
        "energy_mae_kcal_mol": float(np.abs(e_errors).mean()) * KCAL_PER_EV,
        "force_mae_eV_A": float(np.abs(force_flat).mean()),
        "force_rmse_eV_A": float(np.sqrt((force_flat ** 2).mean())),
        "force_mae_kcal_mol_A": float(np.abs(force_flat).mean()) * KCAL_PER_EV,
        "note": (
            "Energies are mean-centred before MAE to cancel the DFT-reference "
            "offset.  Forces are evaluated without centering."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(
    mace_results: dict,
    uma_summary: dict | None,
    hydragnn_summary: dict | None,
):
    KCAL = KCAL_PER_EV
    hdr = (
        f"\n{'Model':<30s}  {'E-MAE (eV)':>10s}  {'E-MAE (kcal/mol)':>17s}"
        f"  {'F-MAE (eV/Å)':>13s}  {'F-MAE (kcal/(mol·Å))':>21s}"
    )
    sep = "-" * 97

    print("\n" + "=" * 97)
    print("  MD17 Uracil MLIP — MACE / UMA / HydraGNN comparison")
    print("=" * 97)
    print(hdr)
    print(sep)

    def _row(label, e_mae, f_mae):
        print(
            f"{label:<30s}"
            f"  {e_mae:10.4f}"
            f"  {e_mae * KCAL:17.4f}"
            f"  {f_mae:13.4f}"
            f"  {f_mae * KCAL:21.4f}"
        )

    # MACE rows
    r = mace_results.get("testset", {})
    for mid in _ALL_MODEL_IDS:
        if mid not in mace_results:
            continue
        sr = mace_results[mid].get("testset", {})
        if sr:
            _row(MACE_MODELS[mid]["label"], sr["energy_mae_eV"], sr["force_mae_eV_A"])

    # UMA row (from uma_benchmark_summary.json)
    if uma_summary:
        ur = uma_summary.get("testset", {})
        if ur:
            model_name = uma_summary.get("model_name", "uma-s-1p2")
            _row(
                f"UMA {model_name}",
                ur.get("energy_mae_eV", float("nan")),
                ur.get("force_mae_eV_A", float("nan")),
            )

    # HydraGNN rows
    if hydragnn_summary:
        labels = {
            "frozen": "HydraGNN (frozen FT)",
            "unfrozen": "HydraGNN (unfrozen FT)",
            "scratch": "HydraGNN (scratch)",
            "ani1x_recycled": "HydraGNN (ANI1x head)",
        }
        for key, label in labels.items():
            if key not in hydragnn_summary:
                continue
            s = hydragnn_summary[key]
            e_mae = s.get("best_energy_mae_eV", float("nan"))
            f_mae = s.get("best_force_mae_eV_A", float("nan"))
            _row(label, e_mae, f_mae)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_ddp()

    parser = argparse.ArgumentParser(
        description="Evaluate MACE models on MD17 and compare to HydraGNN / UMA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=_ALL_MODEL_IDS,
        choices=_ALL_MODEL_IDS,
        help="MACE model ID(s) to evaluate.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device string (cuda / cpu).  Auto-detected if not set.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["testset"],
        help="Dataset split label(s) to evaluate.",
    )
    parser.add_argument(
        "--hydragnn-summary", default=None,
        help="Path to benchmark_summary.json from run_benchmark.py.",
    )
    parser.add_argument(
        "--uma-summary", default=None,
        help="Path to uma_benchmark_summary.json from run_uma_benchmark.py.",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Directory where the MACE results JSON is written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nDataset    : {DATASET_DIR}")
    print(f"Models     : {', '.join(args.models)}")

    results: dict = {}
    out_path = os.path.join(args.output_dir, "mace_benchmark_summary.json")

    for model_id in args.models:
        cfg = MACE_MODELS[model_id]
        print(f"\n{'=' * 60}")
        print(f"  Model : {cfg['label']}")
        print(f"  Data  : {cfg['training_data']}  ({cfg['level_of_theory']})")
        print(f"{'=' * 60}")

        model_results: dict = {"label": cfg["label"]}
        try:
            for split in args.splits:
                print(f"\n--- Evaluating split: {split} ---")
                metrics = evaluate_split(
                    split_label=split,
                    model_id=model_id,
                    device=args.device,
                    verbose=True,
                )
                model_results[split] = metrics
                if metrics:
                    print(
                        f"  Energy MAE : {metrics['energy_mae_eV']:.4f} eV  "
                        f"({metrics['energy_mae_kcal_mol']:.4f} kcal/mol)"
                    )
                    print(
                        f"  Force  MAE : {metrics['force_mae_eV_A']:.4f} eV/Å  "
                        f"({metrics['force_mae_kcal_mol_A']:.4f} kcal/(mol·Å))"
                    )
        except Exception as exc:
            print(f"\n  [SKIP] {cfg['label']} failed: {exc}\n")
            model_results["error"] = str(exc)
        results[model_id] = model_results
        # Save incrementally so partial results are preserved if a later model fails.
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # Load optional comparison summaries
    hydragnn_summary = None
    if args.hydragnn_summary and os.path.isfile(args.hydragnn_summary):
        with open(args.hydragnn_summary) as f:
            hydragnn_summary = json.load(f)

    uma_summary = None
    if args.uma_summary and os.path.isfile(args.uma_summary):
        with open(args.uma_summary) as f:
            uma_summary = json.load(f)

    print_comparison_table(results, uma_summary, hydragnn_summary)


if __name__ == "__main__":
    main()
