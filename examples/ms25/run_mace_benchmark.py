#!/usr/bin/env python3
"""MS25 MACE state-of-the-art benchmark (periodic bulk systems).

Evaluates MACE foundation models zero-shot on the MS25 test splits.  MS25 is a
collection of periodic inorganic MD trajectories, so the default model is the
materials foundation model **MACE-MP-0** (``mace_mp0_medium``, trained on the
Materials Project trajectories at the PBE level).  The organic MACE models
(MACE-OFF / MACE-POLAR / MACE-MH-1) remain selectable via ``--models`` for
completeness.

Dataset
-------
One pickle per system (``dataset/<system>_mlip_peratom.pickle``), produced by
``examples/ms25/ms25_preonly.py``.  Data objects carry total energy (eV),
positions, lattice ``cell``, ``pbc`` flags, and (for VASP systems) forces
(eV/Å).

Energy reference treatment
--------------------------
Energies are mean-centred **per system** (fixed composition across MD frames)
to cancel the constant DFT-reference offset.  Forces are conservative and
evaluated without centering.

Usage
-----
    python examples/ms25/run_mace_benchmark.py
    python examples/ms25/run_mace_benchmark.py --systems MgO-2x2 --models mace_mp0_medium
    python examples/ms25/run_mace_benchmark.py --device cuda \
        --uma-summary examples/ms25/benchmark_results/uma_benchmark_summary.json

Prerequisites
-------------
    pip install mace-torch>=0.3.16
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

from utils.mace_calculator import MACE_MODELS, build_mace_calculator, pyg_data_to_ase_atoms
from utils.finetune_utils import print_timing_summary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KCAL_PER_EV = 23.0609
PICKLE_TAG = "mlip_peratom"
DATASET_ROOT = REPO_ROOT / "dataset"
OUTPUT_DIR = str(REPO_ROOT / "examples" / "ms25" / "benchmark_results")

MS25_SYSTEMS = [
    "MgO-2x2", "MgO-4x4", "H2O-64", "H2O-192",
    "CHA", "HEA", "Reaction", "Zr-O",
]

# Materials-appropriate MACE model for periodic bulk systems.
_DEFAULT_MODELS = ["mace_mp0_medium"]
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


def _dataset_dir(system: str) -> Path:
    return DATASET_ROOT / f"{system}_{PICKLE_TAG}.pickle"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    system: str,
    split_label: str,
    model_id: str,
    calc,
    verbose: bool = True,
) -> dict:
    basedir = str(_dataset_dir(system))
    if not os.path.isdir(basedir):
        print(f"  [{system}/{split_label}] dataset missing ({basedir}) — skipping.")
        return {}

    dataset = SimplePickleDataset(
        basedir=basedir, label=split_label, var_config=_VAR_CONFIG
    )
    if len(dataset) == 0:
        print(f"  [{system}/{split_label}] empty — skipping.")
        return {}

    needs_cs = MACE_MODELS[model_id]["needs_charge_spin"]

    e_preds, e_trues = [], []
    force_errors_flat = []
    n_atoms_total = 0
    have_forces = True
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=True)
        if needs_cs:
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
        atoms.calc = calc

        e_preds.append(float(atoms.get_potential_energy()))
        e_trues.append(float(data.energy.detach().cpu().squeeze()))
        n_atoms_total += len(atoms)

        f_true_tensor = getattr(data, "forces", None)
        if f_true_tensor is None:
            f_true_tensor = getattr(data, "force", None)
        if have_forces and f_true_tensor is not None:
            f_pred = atoms.get_forces()
            f_true = f_true_tensor.detach().cpu().numpy()
            force_errors_flat.append((f_pred - f_true).ravel())
        else:
            have_forces = False

        if verbose and (i + 1) % 50 == 0:
            print(f"    [{system}/{split_label}] {i + 1}/{len(dataset)} done …")

    _inference_sec = time.perf_counter() - _t0
    e_preds = np.asarray(e_preds)
    e_trues = np.asarray(e_trues)
    e_errors = (e_preds - e_preds.mean()) - (e_trues - e_trues.mean())
    n_struct = len(e_errors)
    mean_natoms = n_atoms_total / max(n_struct, 1)

    result = {
        "n_structures": n_struct,
        "mean_natoms": round(mean_natoms, 2),
        "inference_wall_sec": round(_inference_sec, 2),
        "energy_mae_eV": float(np.abs(e_errors).mean()),
        "energy_rmse_eV": float(np.sqrt((e_errors ** 2).mean())),
        "energy_mae_meV_per_atom": float(np.abs(e_errors).mean()) / mean_natoms * 1000.0,
        "note": (
            "Energies mean-centred per system to cancel the DFT-reference "
            "offset; forces evaluated without centering (conservative)."
        ),
    }
    if force_errors_flat:
        force_flat = np.concatenate(force_errors_flat)
        result["force_mae_eV_A"] = float(np.abs(force_flat).mean())
        result["force_rmse_eV_A"] = float(np.sqrt((force_flat ** 2).mean()))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_ddp()

    parser = argparse.ArgumentParser(
        description="Evaluate MACE models on MS25 periodic systems (zero-shot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--systems", nargs="+", default=MS25_SYSTEMS,
                        help="MS25 system name(s) to evaluate.")
    parser.add_argument("--models", nargs="+", default=_DEFAULT_MODELS,
                        choices=_ALL_MODEL_IDS,
                        help="MACE model ID(s).  Default: MACE-MP-0 (materials).")
    parser.add_argument("--device", default=None,
                        help="Device string (cuda / cpu).  Auto-detected if unset.")
    parser.add_argument("--splits", nargs="+", default=["testset"],
                        help="Dataset split label(s) to evaluate.")
    parser.add_argument("--uma-summary", default=None,
                        help="Optional path to uma_benchmark_summary.json.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory where the MACE results JSON is written.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nSystems : {', '.join(args.systems)}")
    print(f"Models  : {', '.join(args.models)}")

    results: dict = {}
    out_path = os.path.join(args.output_dir, "mace_benchmark_summary.json")

    for model_id in args.models:
        cfg = MACE_MODELS[model_id]
        print(f"\n{'#' * 60}\n  MACE model : {cfg['label']}"
              f"  ({cfg['training_data']}, {cfg['level_of_theory']})\n{'#' * 60}")
        calc = build_mace_calculator(model_id=model_id, device=args.device)
        model_results: dict = {"label": cfg["label"]}
        for system in args.systems:
            print(f"\n=== System : {system} ===")
            system_results: dict = {}
            try:
                for split in args.splits:
                    metrics = evaluate_split(system, split, model_id, calc, verbose=True)
                    system_results[split] = metrics
                    if metrics:
                        msg = (
                            f"  [{split}] E-MAE = {metrics['energy_mae_eV']:.4f} eV "
                            f"({metrics['energy_mae_meV_per_atom']:.2f} meV/atom)"
                        )
                        if "force_mae_eV_A" in metrics:
                            msg += f"  F-MAE = {metrics['force_mae_eV_A']:.4f} eV/Å"
                        print(msg)
            except Exception as exc:
                print(f"  [SKIP] {system} failed: {exc}")
                system_results["error"] = str(exc)
            model_results[system] = system_results
        results[model_id] = model_results
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    timing_entries: list[tuple] = []
    for model_id in args.models:
        for system in args.systems:
            r = results.get(model_id, {}).get(system, {}).get("testset", {})
            if r:
                timing_entries.append(
                    (f"{MACE_MODELS[model_id]['label']} — {system}",
                     None, r.get("inference_wall_sec"))
                )
    print_timing_summary(timing_entries)


if __name__ == "__main__":
    main()
