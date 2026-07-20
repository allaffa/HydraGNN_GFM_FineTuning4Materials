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

def _composition_key(data) -> tuple:
    """Return a hashable composition fingerprint (element, count) tuple.

    Used as a molecule identity proxy when no mol_id is stored.  Structures
    with identical composition are assumed to be conformers of the same molecule.
    """
    import torch
    z = data.x[:, 0].long()
    unique, counts = torch.unique(z, return_counts=True)
    return tuple(zip(unique.tolist(), counts.tolist()))


def evaluate_split(
    split_label: str,
    uma_model: str,
    uma_task: str,
    device: str | None,
    verbose: bool = True,
) -> dict:
    """Load a dataset split and compute UMA conformational energy MAE.

    Wiggle150 stores energies as ``E_conformer - E_minimum_conformer`` (relative
    to the lowest-energy structure of each molecule).  UMA predicts absolute
    total energies.  To align the two scales we:

      1. Group structures by composition fingerprint (proxy for molecule identity).
      2. Within each group subtract the minimum UMA total energy, obtaining
         UMA-predicted conformational energies relative to the same reference
         convention as the GT.
      3. Compute MAE between UMA relative energies and GT relative energies.
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

    # Collect per-structure results indexed by composition group
    from collections import defaultdict
    groups: dict = defaultdict(lambda: {"uma": [], "gt": [], "idx": []})

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = calc

        e_uma = float(atoms.get_potential_energy())

        y = data.y.detach().cpu()
        e_gt = float(y.item() if y.dim() == 0 else y.view(-1)[0])

        comp_key = _composition_key(data)
        groups[comp_key]["uma"].append(e_uma)
        groups[comp_key]["gt"].append(e_gt)
        groups[comp_key]["idx"].append(i)

        if verbose and (i + 1) % 100 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    # Compute relative UMA energies within each composition group and collect errors
    errors = []
    for comp_key, grp in groups.items():
        uma_arr = np.asarray(grp["uma"])
        gt_arr = np.asarray(grp["gt"])
        # Shift UMA to same reference as GT (minimum = 0 within the group)
        uma_rel = uma_arr - uma_arr.min()
        errors.extend((uma_rel - gt_arr).tolist())

    errors = np.asarray(errors)

    return {
        "n_structures": len(errors),
        "n_composition_groups": len(groups),
        "energy_mae_eV": float(np.abs(errors).mean()),
        "energy_rmse_eV": float(np.sqrt((errors ** 2).mean())),
        "energy_mae_kcal_mol": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "UMA total energies are made relative within each composition group "
            "(min-shifted) to match the Wiggle150 convention of "
            "E_conf - E_min_conf.  Groups are identified by composition fingerprint."
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
