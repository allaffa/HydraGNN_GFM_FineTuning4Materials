#!/usr/bin/env python3
"""Wiggle150 UMA state-of-the-art benchmark.

Evaluates Meta's UMA (Universal Model for Atoms) on the Wiggle150 test split
used by the HydraGNN fine-tuning experiments and compares the results to the
fine-tuned HydraGNN numbers produced by ``run_benchmark.py``.

Dataset
-------
Wiggle150 pickle (``dataset/wiggle150.pickle``), preprocessed by
``examples/wiggle150/wiggle150_preonly.py``.  Energies are in eV (relative
conformer energies).  The dataset consists of molecular conformers (non-periodic),
so we use the UMA ``omol`` task head.

Note on energy reference
------------------------
Wiggle150 stores *relative* conformer energies (offset to the ground-state
conformer of each molecule).  UMA predicts *total* potential energies on an
absolute scale.  To make the comparison meaningful, we mean-center both the UMA
predictions and the ground-truth values per-molecule before computing MAE — the
same convention as the HydraGNN fine-tuning benchmark.

Usage
-----
    python examples/wiggle150/run_uma_benchmark.py

    python examples/wiggle150/run_uma_benchmark.py \
        --uma-model uma-s-1p2 --uma-task omol

    python examples/wiggle150/run_uma_benchmark.py \
        --hydragnn-summary examples/wiggle150/benchmark_results/benchmark_summary.json

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
DATASET_DIR = str(REPO_ROOT / "dataset" / "wiggle150.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "wiggle150" / "benchmark_results")
KCAL_PER_EV = 23.0609

_VAR_CONFIG = {
    "graph_feature_names": ["energy"],
    "graph_feature_dims": [1],
    "node_feature_names": ["atomic_number"],
    "node_feature_dims": [1],
    "input_node_features": [0],
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
    """Load a dataset split and compute UMA energy MAE (mean-centred per molecule)."""
    dataset = SimplePickleDataset(
        basedir=DATASET_DIR, label=split_label, var_config=_VAR_CONFIG
    )
    if len(dataset) == 0:
        print(f"  [{split_label}] empty — skipping.")
        return {}

    calc = build_uma_calculator(
        model_name=uma_model, task_name=uma_task, device=device
    )

    # Collect raw energies grouped by molecule (identified by mol_id if present,
    # otherwise treated as a single group).
    e_preds = []
    e_trues = []
    mol_ids = []

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        atoms.calc = calc

        e_preds.append(float(atoms.get_potential_energy()))

        # Ground-truth energy stored on the Data object.
        # Wiggle150 stores relative energy in data.y (after var_config mapping)
        # or data.energy depending on preprocessing version.
        if hasattr(data, "energy") and data.energy is not None:
            e_trues.append(float(data.energy.detach().cpu().squeeze()))
        else:
            # Fall back to batch.y[0] graph-level target
            e_trues.append(float(data.y.detach().cpu().squeeze()[0]))

        # Molecule identifier for mean-centering (optional field)
        mol_id = getattr(data, "mol_id", None)
        if mol_id is None:
            mol_id = getattr(data, "mol_prefix", None)
        mol_ids.append(mol_id if mol_id is not None else 0)

        if verbose and (i + 1) % 100 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    e_preds = np.asarray(e_preds)
    e_trues = np.asarray(e_trues)

    # ------------------------------------------------------------------
    # Mean-centre both arrays per molecule so that absolute energy offsets
    # between UMA (ωB97X-D / DFT-MLFF) and Wiggle150 (training DFT level)
    # cancel out.
    # ------------------------------------------------------------------
    unique_ids = list(dict.fromkeys(mol_ids))  # preserve insertion order
    e_preds_centred = e_preds.copy()
    e_trues_centred = e_trues.copy()

    for uid in unique_ids:
        mask = np.array([m == uid for m in mol_ids])
        e_preds_centred[mask] -= e_preds[mask].mean()
        e_trues_centred[mask] -= e_trues[mask].mean()

    errors = e_preds_centred - e_trues_centred

    return {
        "n_structures": len(errors),
        "n_molecules": len(unique_ids),
        "energy_mae_eV": float(np.abs(errors).mean()),
        "energy_rmse_eV": float(np.sqrt((errors ** 2).mean())),
        "energy_mae_kcal_mol": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "Energies are mean-centred per molecule before computing MAE, "
            "to cancel absolute DFT-reference offsets between UMA and "
            "the Wiggle150 training labels."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(uma_results: dict, hydragnn_summary: dict | None):
    print("\n" + "=" * 70)
    print("  Wiggle150 — UMA vs HydraGNN fine-tuning comparison")
    print("=" * 70)

    r = uma_results.get("testset", {})
    if r:
        print(
            f"\n{'Model':<28s}  {'E-MAE (eV)':>10s}  {'E-MAE (kcal/mol)':>17s}"
        )
        print("-" * 60)
        print(
            f"{'UMA ' + uma_results['model_name']:<28s}"
            f"  {r['energy_mae_eV']:10.4f}"
            f"  {r['energy_mae_kcal_mol']:17.4f}"
        )

    if hydragnn_summary:
        labels = {
            "frozen_fp64": "HydraGNN (frozen, fp64)",
            "unfrozen_fp64": "HydraGNN (unfrozen, fp64)",
            "scratch_fp64": "HydraGNN (scratch, fp64)",
            # Fallback keys used by older benchmark scripts
            "frozen": "HydraGNN (frozen FT)",
            "unfrozen": "HydraGNN (unfrozen FT)",
            "scratch": "HydraGNN (scratch)",
        }
        printed = set()
        for key, label in labels.items():
            if key not in hydragnn_summary or key in printed:
                continue
            s = hydragnn_summary[key]
            e_mae = s.get("best_mae_eV", s.get("best_energy_mae_eV", float("nan")))
            print(
                f"{label:<28s}"
                f"  {e_mae:10.4f}"
                f"  {e_mae * KCAL_PER_EV:17.4f}"
            )
            printed.add(key)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_ddp()

    parser = argparse.ArgumentParser(
        description="Evaluate UMA on Wiggle150 and compare to HydraGNN fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--uma-model", default="uma-s-1p2",
        help="UMA model name or local checkpoint path.",
    )
    parser.add_argument(
        "--uma-task", default="omol",
        choices=["omat", "omol", "oc20", "odac", "omc"],
        help="UMA task head.  Wiggle150 molecular conformers → omol.",
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
                f"({metrics['energy_mae_kcal_mol']:.4f} kcal/mol)  "
                f"[mean-centred per molecule]"
            )

    out_path = os.path.join(args.output_dir, "uma_benchmark_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    hydragnn_summary = None
    if args.hydragnn_summary and os.path.isfile(args.hydragnn_summary):
        with open(args.hydragnn_summary) as f:
            hydragnn_summary = json.load(f)

    print_comparison_table(results, hydragnn_summary)


if __name__ == "__main__":
    main()
