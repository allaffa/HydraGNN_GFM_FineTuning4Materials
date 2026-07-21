#!/usr/bin/env python3
"""QM9 UMA state-of-the-art benchmark.

Evaluates Meta's UMA (Universal Model for Atoms) on the QM9 test split used
by the HydraGNN fine-tuning experiments and compares the results to the
fine-tuned HydraGNN numbers produced by ``run_benchmark.py``.

Dataset
-------
QM9 energy pickle (``dataset/qm9_energy.pickle``), preprocessed by
``examples/qm9/qm9_energy_preonly.py``.  The stored target (``data.y``) is the
per-atom atomization energy in eV/atom (QM9 target U0, mean-shifted).  The
dataset consists of small organic molecules (non-periodic), so we use the UMA
``omol`` task head.

Comparison methodology
----------------------
UMA predicts an *absolute* total DFT potential energy, while the QM9 labels are
*atomization* energies (relative to isolated atoms).  To make the comparison
meaningful we:

  1. Compute UMA total energy per molecule and divide by the number of atoms
     to obtain a per-atom energy on the UMA scale.
  2. Mean-centre both the UMA per-atom energies and the QM9 per-atom targets
     over the evaluation split.
  3. Compute MAE on the mean-centred values.

This mirrors the methodology used for the Wiggle150 benchmark and captures how
well UMA discriminates between different molecules (relative ordering), which
is what the fine-tuning benchmark measures after mean-shifting.

Usage
-----
    python examples/qm9/run_uma_benchmark.py

    python examples/qm9/run_uma_benchmark.py \
        --uma-model uma-s-1p2 --uma-task omol

    python examples/qm9/run_uma_benchmark.py \
        --hydragnn-summary examples/qm9/benchmark_results/benchmark_summary.json

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
DATASET_DIR = str(REPO_ROOT / "dataset" / "qm9_energy.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "qm9" / "benchmark_results")
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
# Linear reference energy correction
# ---------------------------------------------------------------------------

def _fit_linear_reference(e_preds_total, compositions, gt_per_atom, n_atoms_list):
    """Fit per-element reference energies (eV/atom of element type) by least squares.

    Finds alpha_Z such that:  E_pred_total - sum_Z(n_Z * alpha_Z) ≈ U0_total
    where U0_total = gt_per_atom * n_atoms.

    Returns (alpha array, sorted element list).
    """
    all_elements = sorted({Z for comp in compositions for Z in comp})
    elem_idx = {Z: j for j, Z in enumerate(all_elements)}
    n = len(e_preds_total)
    A = np.zeros((n, len(all_elements)))
    for i, comp in enumerate(compositions):
        for Z, count in comp.items():
            A[i, elem_idx[Z]] = count
    # b[i] = what the reference term must account for
    b = np.asarray(e_preds_total) - np.asarray(gt_per_atom) * np.asarray(n_atoms_list)
    alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return alpha, all_elements


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
    """Load a dataset split and compute UMA per-atom energy MAE.

    UMA predicts absolute total DFT energies; QM9 labels are per-atom
    atomization energies (reference-subtracted).  A per-element linear
    reference correction is fitted on the evaluation split via least squares
    so that the composition-dependent energy offset is removed before
    computing MAE.  This is the standard zero-shot MLIP evaluation protocol
    for QM9.
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

    e_preds_total = []   # UMA absolute total energy (eV)
    compositions = []    # list of dicts {atomic_number: count}
    n_atoms_list = []    # number of atoms per structure
    gt_per_atom = []     # ground-truth per-atom atomization energy (eV/atom)

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = calc

        n_atoms = len(atoms)
        e_total = float(atoms.get_potential_energy())  # eV (absolute total)
        e_preds_total.append(e_total)
        n_atoms_list.append(n_atoms)

        comp: dict[int, int] = {}
        for Z in atoms.get_atomic_numbers():
            comp[int(Z)] = comp.get(int(Z), 0) + 1
        compositions.append(comp)

        # QM9 ground truth: per-atom atomization energy in eV/atom.
        y = data.y.detach().cpu()
        gt_val = float(y.squeeze()) if y.numel() == 1 else float(y.view(-1)[0])
        gt_per_atom.append(gt_val)

        if verbose and (i + 1) % 100 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    # ------------------------------------------------------------------
    # Fit per-element linear reference energies on this split, then
    # subtract them to align the UMA energy scale with the QM9 U0 target.
    # ------------------------------------------------------------------
    alpha, elements = _fit_linear_reference(
        e_preds_total, compositions, gt_per_atom, n_atoms_list
    )
    elem_idx = {Z: j for j, Z in enumerate(elements)}

    corrected_per_atom = []
    for e_total, comp, n in zip(e_preds_total, compositions, n_atoms_list):
        ref = sum(comp.get(Z, 0) * alpha[elem_idx[Z]] for Z in elements)
        corrected_per_atom.append((e_total - ref) / n)

    pred_arr = np.asarray(corrected_per_atom)
    gt_arr = np.asarray(gt_per_atom)
    errors = pred_arr - gt_arr

    return {
        "n_structures": len(errors),
        "energy_per_atom_mae_eV": float(np.abs(errors).mean()),
        "energy_per_atom_rmse_eV": float(np.sqrt((errors ** 2).mean())),
        "energy_per_atom_mae_kcal_mol": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "UMA total energy corrected by per-element linear reference "
            "energies (least-squares fit on evaluation split), divided by "
            "n_atoms, compared to QM9 per-atom atomization energy."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(uma_results: dict, hydragnn_summary: dict | None):
    print("\n" + "=" * 70)
    print("  QM9 (U0 atomization energy) — UMA vs HydraGNN comparison")
    print("=" * 70)
    print("  [Metric: per-atom energy MAE (eV/atom), linear ref. correction]")

    r = uma_results.get("testset", {})
    if r:
        print(
            f"\n{'Model':<28s}  {'E/atom MAE (eV)':>15s}  {'E/atom MAE (kcal/mol)':>22s}"
        )
        print("-" * 70)
        print(
            f"{'UMA ' + uma_results['model_name']:<28s}"
            f"  {r['energy_per_atom_mae_eV']:15.4f}"
            f"  {r['energy_per_atom_mae_kcal_mol']:22.4f}"
        )

    if hydragnn_summary:
        labels = {
            "frozen": "HydraGNN (frozen FT)",
            "unfrozen": "HydraGNN (unfrozen FT)",
            "scratch": "HydraGNN (scratch)",
            "ani1x_recycled": "HydraGNN (ANI1x head)",
            "qm7x_recycled": "HydraGNN (QM7x head)",
        }
        for key, label in labels.items():
            if key not in hydragnn_summary:
                continue
            s = hydragnn_summary[key]
            e_mae = s.get(
                "best_energy_per_atom_mae_eV",
                s.get(
                    "best_energy_mae_eV_atom",
                    s.get("best_mae_eV", s.get("best_energy_mae_eV", float("nan"))),
                ),
            )
            print(
                f"{label:<28s}"
                f"  {e_mae:15.4f}"
                f"  {e_mae * KCAL_PER_EV:22.4f}"
            )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_ddp()

    parser = argparse.ArgumentParser(
        description="Evaluate UMA on QM9 and compare to HydraGNN fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--uma-model", default="uma-s-1p2",
        help="UMA model name or local checkpoint path.",
    )
    parser.add_argument(
        "--uma-task", default="omol",
        choices=["omat", "omol", "oc20", "odac", "omc"],
        help="UMA task head.  QM9 molecules → omol.",
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
                f"  Energy/atom MAE : {metrics['energy_per_atom_mae_eV']:.4f} eV/atom  "
                f"({metrics['energy_per_atom_mae_kcal_mol']:.4f} kcal/mol)  "
                f"[linear ref. correction]"
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
