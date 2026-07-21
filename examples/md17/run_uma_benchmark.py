#!/usr/bin/env python3
"""MD17 UMA state-of-the-art benchmark.

Evaluates Meta's UMA (Universal Model for Atoms) on the MD17 test split used
by the HydraGNN fine-tuning experiments and compares the results to the
fine-tuned HydraGNN numbers produced by ``run_benchmark.py``.

Dataset
-------
MD17 uracil MLIP pickle (``dataset/md17_mlip.pickle``), preprocessed by
``examples/md17/md17_mlip_preonly.py``.  Energies are in eV; forces in eV/Å.
These are single-molecule trajectories (non-periodic), so we use the UMA
``omol`` task head.

Usage
-----
    # Single-GPU / CPU
    python examples/md17/run_uma_benchmark.py

    # Override model or task
    python examples/md17/run_uma_benchmark.py \
        --uma-model uma-s-1p2 --uma-task omol

    # Compare against a previously saved HydraGNN benchmark JSON
    python examples/md17/run_uma_benchmark.py \
        --hydragnn-summary examples/md17/benchmark_results/benchmark_summary.json

Prerequisites
-------------
    pip install fairchem-core>=2.20
    huggingface-cli login   # accept UMA license at huggingface.co/facebook/UMA

Reference
---------
Wood et al., "UMA: A Family of Universal Models for Atoms", arXiv:2506.23971
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

from utils.uma_calculator import build_uma_calculator, pyg_data_to_ase_atoms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR = str(REPO_ROOT / "dataset" / "md17_mlip.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "md17" / "benchmark_results")
KCAL_PER_EV = 23.0609

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
    uma_model: str,
    uma_task: str,
    device: str | None,
    verbose: bool = True,
) -> dict:
    """Load a dataset split and compute UMA energy + force MAE.

    Energies are mean-centred before computing MAE to cancel the absolute
    DFT-energy-reference offset between UMA (ωB97X-D) and the MD17 training
    labels — consistent with the HydraGNN fine-tuning benchmark that also
    trains on mean-shifted energies.  Forces are reference-invariant and are
    evaluated without any centering.
    """
    dataset = SimplePickleDataset(
        basedir=DATASET_DIR, label=split_label, var_config=_VAR_CONFIG
    )
    if len(dataset) == 0:
        print(f"  [{split_label}] empty — skipping.")
        return {}

    calc = build_uma_calculator(
        model_name=uma_model, task_name=uma_task, device=device
    )

    e_preds = []
    e_trues = []
    force_errors_flat = []
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        # Set charge/spin explicitly to silence fairchem warnings.
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = calc

        e_preds.append(float(atoms.get_potential_energy()))
        f_pred = atoms.get_forces()  # [N, 3], eV/Å

        e_trues.append(float(data.energy.detach().cpu().squeeze()))
        f_true = data.forces.detach().cpu().numpy()  # [N, 3]

        force_errors_flat.append((f_pred - f_true).ravel())

        if verbose and (i + 1) % 100 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    _inference_sec = time.perf_counter() - _t0
    e_preds = np.asarray(e_preds)
    e_trues = np.asarray(e_trues)

    # Mean-centre both energy arrays to cancel the DFT-reference offset.
    e_preds_c = e_preds - e_preds.mean()
    e_trues_c = e_trues - e_trues.mean()
    energy_errors = e_preds_c - e_trues_c

    force_flat = np.concatenate(force_errors_flat)

    return {
        "n_structures": len(energy_errors),
        "inference_wall_sec": round(_inference_sec, 2),
        "energy_mae_eV": float(np.abs(energy_errors).mean()),
        "energy_rmse_eV": float(np.sqrt((energy_errors ** 2).mean())),
        "energy_mae_kcal_mol": float(np.abs(energy_errors).mean()) * KCAL_PER_EV,
        "force_mae_eV_A": float(np.abs(force_flat).mean()),
        "force_rmse_eV_A": float(np.sqrt((force_flat ** 2).mean())),
        "force_mae_kcal_mol_A": float(np.abs(force_flat).mean()) * KCAL_PER_EV,
        "note": (
            "Energies are mean-centred before MAE to cancel the DFT-reference "
            "offset between UMA and the MD17 training labels. "
            "Forces are evaluated without centering (reference-invariant)."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(uma_results: dict, hydragnn_summary: dict | None):
    KCAL = KCAL_PER_EV
    print("\n" + "=" * 80)
    print("  MD17 Uracil MLIP — UMA vs HydraGNN fine-tuning comparison")
    print("=" * 80)

    # UMA row
    r = uma_results.get("testset", {})
    if r:
        print(
            f"\n{'Model':<22s}  {'E-MAE (eV)':>10s}  {'E-MAE (kcal/mol)':>17s}"
            f"  {'F-MAE (eV/Å)':>13s}  {'F-MAE (kcal/(mol·Å))':>21s}"
        )
        print("-" * 90)
        print(
            f"{'UMA ' + uma_results['model_name']:<22s}"
            f"  {r['energy_mae_eV']:10.4f}"
            f"  {r['energy_mae_kcal_mol']:17.4f}"
            f"  {r['force_mae_eV_A']:13.4f}"
            f"  {r['force_mae_kcal_mol_A']:21.4f}"
        )

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
            print(
                f"{label:<22s}"
                f"  {e_mae:10.4f}"
                f"  {e_mae * KCAL:17.4f}"
                f"  {f_mae:13.4f}"
                f"  {f_mae * KCAL:21.4f}"
            )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # DDP is not required for pure-inference, but keep setup for consistency
    # with the rest of the benchmarks (single process is fine).
    setup_ddp()

    parser = argparse.ArgumentParser(
        description="Evaluate UMA on MD17 and compare to HydraGNN fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--uma-model", default="uma-s-1p2",
        help="UMA model name (uma-s-1p2 / uma-s-1p1 / uma-m-1p1) or local checkpoint path.",
    )
    parser.add_argument(
        "--uma-task", default="omol",
        choices=["omat", "omol", "oc20", "odac", "omc"],
        help="UMA task head.  MD17 molecules → omol.",
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
        help="Path to benchmark_summary.json from run_benchmark.py for side-by-side table.",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Directory where the UMA results JSON is written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nUMA model  : {args.uma_model}")
    print(f"UMA task   : {args.uma_task}")
    print(f"Dataset    : {DATASET_DIR}")

    results: dict = {
        "model_name": args.uma_model,
        "task_name": args.uma_task,
        "dataset": DATASET_DIR,
    }

    for split in args.splits:
        print(f"\n--- Evaluating split: {split} ---")
        metrics = evaluate_split(
            split_label=split,
            uma_model=args.uma_model,
            uma_task=args.uma_task,
            device=args.device,
            verbose=True,
        )
        results[split] = metrics
        if metrics:
            print(
                f"  Energy MAE : {metrics['energy_mae_eV']:.4f} eV  "
                f"({metrics['energy_mae_kcal_mol']:.4f} kcal/mol)"
            )
            print(
                f"  Force  MAE : {metrics['force_mae_eV_A']:.4f} eV/Å  "
                f"({metrics['force_mae_kcal_mol_A']:.4f} kcal/(mol·Å))"
            )

    # Save results
    out_path = os.path.join(args.output_dir, "uma_benchmark_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Load HydraGNN comparison summary if provided
    hydragnn_summary = None
    if args.hydragnn_summary and os.path.isfile(args.hydragnn_summary):
        with open(args.hydragnn_summary) as f:
            hydragnn_summary = json.load(f)

    print_comparison_table(results, hydragnn_summary)


if __name__ == "__main__":
    main()
