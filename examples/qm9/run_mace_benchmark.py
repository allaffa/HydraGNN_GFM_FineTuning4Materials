#!/usr/bin/env python3
"""QM9 MACE state-of-the-art benchmark.

Evaluates three MACE foundation models (MACE-OFF23 medium, MACE-POLAR-1
polar-1-m, MACE-MH-1 omol head) on the QM9 test split and compares the
results to the HydraGNN fine-tuning numbers from ``run_benchmark.py`` and
optionally the UMA numbers from ``run_uma_benchmark.py``.

Dataset
-------
QM9 energy pickle (``dataset/qm9_energy.pickle``), preprocessed by
``examples/qm9/qm9_energy_preonly.py``.  The stored target (``data.y``) is the
per-atom atomization energy in eV/atom (QM9 target U0, mean-shifted).
Non-periodic small organic molecules.

Energy reference treatment
--------------------------
MACE predicts *absolute* total DFT energies while QM9 labels are *atomization*
energies.  To make the comparison meaningful we apply a per-element linear
reference energy correction:

  1. Compute MACE total energy per molecule.
  2. Fit per-element reference energies alpha_Z by least squares on the
     evaluation split:  E_MACE - sum_Z(n_Z * alpha_Z) ≈ U0_total
  3. Subtract the fitted reference term and divide by n_atoms.
  4. Compare corrected per-atom energies to the QM9 U0 per-atom target.

This is the standard zero-shot MLIP evaluation protocol for QM9.

Usage
-----
    python examples/qm9/run_mace_benchmark.py

    # Select a subset of models
    python examples/qm9/run_mace_benchmark.py \\
        --models mace_off_medium mace_mh1_omol

    # Full comparison table (MACE + UMA + HydraGNN)
    python examples/qm9/run_mace_benchmark.py \\
        --hydragnn-summary examples/qm9/benchmark_results/benchmark_summary.json \\
        --uma-summary      examples/qm9/benchmark_results/uma_benchmark_summary.json

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
DATASET_DIR = str(REPO_ROOT / "dataset" / "qm9_energy.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "qm9" / "benchmark_results")
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
    b = np.asarray(e_preds_total) - np.asarray(gt_per_atom) * np.asarray(n_atoms_list)
    alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return alpha, all_elements


def evaluate_split(
    split_label: str,
    model_id: str,
    device: str | None,
    verbose: bool = True,
) -> dict:
    """Evaluate a single MACE model on one QM9 split.

    MACE predicts absolute total DFT energies; QM9 labels are per-atom
    atomization energies.  A per-element linear reference correction is
    fitted on the evaluation split via least squares to align energy scales
    before computing MAE.  This is the standard zero-shot MLIP protocol.
    """
    dataset = SimplePickleDataset(
        basedir=DATASET_DIR, label=split_label, var_config=_VAR_CONFIG
    )
    if len(dataset) == 0:
        print(f"  [{split_label}] empty — skipping.")
        return {}

    calc = build_mace_calculator(model_id=model_id, device=device)
    needs_cs = MACE_MODELS[model_id]["needs_charge_spin"]

    e_preds_total = []
    compositions = []
    n_atoms_list = []
    gt_per_atom = []
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        if needs_cs:
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
        atoms.calc = calc

        n_atoms = len(atoms)
        e_total = float(atoms.get_potential_energy())
        e_preds_total.append(e_total)
        n_atoms_list.append(n_atoms)

        comp: dict[int, int] = {}
        for Z in atoms.get_atomic_numbers():
            comp[int(Z)] = comp.get(int(Z), 0) + 1
        compositions.append(comp)

        y = data.y.detach().cpu()
        gt_val = float(y.squeeze()) if y.numel() == 1 else float(y.view(-1)[0])
        gt_per_atom.append(gt_val)

        if verbose and (i + 1) % 100 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    _inference_sec = time.perf_counter() - _t0
    # Fit per-element linear reference energies, then subtract to align
    # the MACE total energy scale with the QM9 U0 atomization target.
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
            "MACE total energy corrected by per-element linear reference "
            "energies (least-squares fit on evaluation split), divided by "
            "n_atoms, compared to QM9 per-atom atomization energy."
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
        f"\n{'Model':<30s}  {'E/atom MAE (eV)':>15s}  {'E/atom MAE (kcal/mol)':>22s}"
    )
    sep = "-" * 72

    print("\n" + "=" * 72)
    print("  QM9 (U0 atomization energy) — MACE / UMA / HydraGNN comparison")
    print("  [Metric: per-atom energy MAE (eV/atom), linear ref. correction]")
    print("=" * 72)
    print(hdr)
    print(sep)

    def _row(label, e_mae):
        print(f"{label:<30s}  {e_mae:15.4f}  {e_mae * KCAL:22.4f}")

    # MACE rows
    for mid in _ALL_MODEL_IDS:
        if mid not in mace_results:
            continue
        sr = mace_results[mid].get("testset", {})
        if sr:
            _row(MACE_MODELS[mid]["label"], sr["energy_per_atom_mae_eV"])

    # UMA row
    if uma_summary:
        ur = uma_summary.get("testset", {})
        if ur:
            model_name = uma_summary.get("model_name", "uma-s-1p2")
            _row(
                f"UMA {model_name}",
                ur.get(
                    "energy_per_atom_mae_eV",
                    ur.get("energy_mae_eV", float("nan")),
                ),
            )

    # HydraGNN rows
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
            _row(label, e_mae)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_ddp()

    parser = argparse.ArgumentParser(
        description="Evaluate MACE models on QM9 and compare to HydraGNN / UMA.",
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
                        f"  Energy/atom MAE : {metrics['energy_per_atom_mae_eV']:.4f} eV/atom  "
                        f"({metrics['energy_per_atom_mae_kcal_mol']:.4f} kcal/mol/atom)"
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
