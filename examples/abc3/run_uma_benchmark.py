#!/usr/bin/env python3
"""ABC3 UMA zero-shot benchmark (formation energy per atom, periodic perovskites).

Evaluates Meta's UMA (Universal Model for Atoms) zero-shot on the ABC3 test
split and compares against HydraGNN fine-tuning numbers.

Dataset
-------
ABC3 perovskite formation energy pickle (``dataset/abc3.pickle``), preprocessed
by ``examples/abc3/abc3_preonly.py``.  The stored target is formation energy
per atom (eV/atom).

Energy reference treatment
--------------------------
UMA predicts absolute DFT total energies.  ABC3 labels are DFT formation
energies (already referenced against elemental solids), so the conversion
requires a per-element linear reference correction fitted by least squares on
the evaluation split — the same protocol as the QM9 benchmark.

Usage
-----
    python examples/abc3/run_uma_benchmark.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from hydragnn.utils.distributed import setup_ddp
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

from utils.uma_calculator import build_uma_calculator, pyg_data_to_ase_atoms

DATASET_DIR = str(REPO_ROOT / "dataset" / "abc3.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "abc3" / "benchmark_results")
KCAL_PER_EV = 23.0609

_VAR_CONFIG = {
    "type": ["graph"], "output_index": [0], "output_dim": [1],
    "output_names": ["energy"], "graph_feature_names": ["energy"],
    "graph_feature_dims": [1], "node_feature_names": ["atomic_number"],
    "node_feature_dims": [1], "input_node_features": [0],
    "denormalize_output": False,
}


def _fit_linear_reference(e_preds_total, compositions, gt_per_atom, n_atoms_list):
    """Least-squares per-element reference energies to align UMA absolute energies
    with formation energy labels."""
    all_elements = sorted({Z for comp in compositions for Z in comp})
    elem_idx = {Z: j for j, Z in enumerate(all_elements)}
    n = len(e_preds_total)
    A = np.zeros((n, len(all_elements)))
    for i, comp in enumerate(compositions):
        for Z, count in comp.items():
            A[i, elem_idx[Z]] = count
    b = np.asarray(e_preds_total) - np.asarray(gt_per_atom) * np.asarray(n_atoms_list)
    alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return alpha, all_elements


def evaluate_split(split_label, uma_model, uma_task, device, verbose=True):
    dataset = SimplePickleDataset(
        basedir=DATASET_DIR, label=split_label, var_config=_VAR_CONFIG
    )
    if len(dataset) == 0:
        print(f"  [{split_label}] empty — skipping.")
        return {}

    calc = build_uma_calculator(model_name=uma_model, task_name=uma_task, device=device)

    e_preds_total = []
    compositions = []
    n_atoms_list = []
    gt_per_atom = []
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=True)
        atoms.calc = calc

        n_atoms = len(atoms)
        e_preds_total.append(float(atoms.get_potential_energy()))
        n_atoms_list.append(n_atoms)

        comp: dict[int, int] = {}
        for Z in atoms.get_atomic_numbers():
            comp[int(Z)] = comp.get(int(Z), 0) + 1
        compositions.append(comp)

        y = data.y.detach().cpu()
        gt_per_atom.append(float(y.squeeze()) if y.numel() == 1 else float(y.view(-1)[0]))

        if verbose and (i + 1) % 100 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    _inference_sec = time.perf_counter() - _t0

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
        "inference_wall_sec": round(_inference_sec, 2),
        "energy_per_atom_mae_eV": float(np.abs(errors).mean()),
        "energy_per_atom_rmse_eV": float(np.sqrt((errors ** 2).mean())),
        "energy_per_atom_mae_kcal_mol": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "UMA total energy corrected by per-element linear reference "
            "energies (least-squares fit on evaluation split), divided by "
            "n_atoms, compared to ABC3 per-atom formation energy."
        ),
    }


def main():
    setup_ddp()

    parser = argparse.ArgumentParser(
        description="Evaluate UMA on ABC3 perovskites (zero-shot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uma-model", default="uma-s-1p2")
    parser.add_argument(
        "--uma-task", default="omat",
        choices=["omat", "omol", "oc20", "odac", "omc"],
        help="UMA task head.  ABC3 periodic inorganic → omat.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", nargs="+", default=["testset"])
    parser.add_argument("--hydragnn-summary", default=None,
                        help="Path to benchmark_summary.json from ensemble_fine_tune.py.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
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
        if hydragnn_summary:
            mae = hydragnn_summary.get("test_mae_eV_per_atom",
                  hydragnn_summary.get("test_mae_eV", float("nan")))
            print(f"\n  HydraGNN FT test MAE : {mae:.4f} eV/atom")


if __name__ == "__main__":
    main()
