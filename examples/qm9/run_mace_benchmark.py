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
energies.  To make the comparison meaningful we:

  1. Compute MACE total energy per molecule and divide by the number of atoms.
  2. Mean-centre both MACE per-atom energies and QM9 per-atom targets over the
     evaluation split so that the absolute reference offset cancels.
  3. Compute MAE on the mean-centred values.

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

def evaluate_split(
    split_label: str,
    model_id: str,
    device: str | None,
    verbose: bool = True,
) -> dict:
    """Evaluate a single MACE model on one QM9 split.

    MACE total energy / n_atoms is mean-centred against the QM9 per-atom
    atomization energy (mean-centred) to cancel absolute reference offsets.
    """
    dataset = SimplePickleDataset(
        basedir=DATASET_DIR, label=split_label, var_config=_VAR_CONFIG
    )
    if len(dataset) == 0:
        print(f"  [{split_label}] empty — skipping.")
        return {}

    calc = build_mace_calculator(model_id=model_id, device=device)
    needs_cs = MACE_MODELS[model_id]["needs_charge_spin"]

    mace_per_atom = []
    gt_per_atom = []

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        if needs_cs:
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
        atoms.calc = calc

        n_atoms = len(atoms)
        e_total = float(atoms.get_potential_energy())
        mace_per_atom.append(e_total / n_atoms)

        y = data.y.detach().cpu()
        gt_val = float(y.squeeze()) if y.numel() == 1 else float(y.view(-1)[0])
        gt_per_atom.append(gt_val)

        if verbose and (i + 1) % 100 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    mace_arr = np.asarray(mace_per_atom)
    gt_arr = np.asarray(gt_per_atom)

    errors = (mace_arr - mace_arr.mean()) - (gt_arr - gt_arr.mean())

    return {
        "n_structures": len(errors),
        "energy_per_atom_mae_eV": float(np.abs(errors).mean()),
        "energy_per_atom_rmse_eV": float(np.sqrt((errors ** 2).mean())),
        "energy_per_atom_mae_kcal_mol": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "MACE total energy / n_atoms, mean-centred against the QM9 "
            "per-atom atomization energy (mean-centred)."
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
    print("  [Metric: per-atom energy MAE (eV/atom), mean-centred]")
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

    for model_id in args.models:
        cfg = MACE_MODELS[model_id]
        print(f"\n{'=' * 60}")
        print(f"  Model : {cfg['label']}")
        print(f"  Data  : {cfg['training_data']}  ({cfg['level_of_theory']})")
        print(f"{'=' * 60}")

        model_results: dict = {"label": cfg["label"]}
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
        results[model_id] = model_results

    # Save results
    out_path = os.path.join(args.output_dir, "mace_benchmark_summary.json")
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
