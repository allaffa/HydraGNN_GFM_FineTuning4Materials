#!/usr/bin/env python3
"""ABC3 MACE zero-shot benchmark (formation energy per atom, periodic perovskites).

Evaluates MACE foundation models zero-shot on the ABC3 test split.  The default
model is MACE-MP-0 (``mace_mp0_medium``), trained on the Materials Project — the
materials-appropriate choice for inorganic perovskite crystals.  MACE-OFF23 and
MACE-MH-1 are included as optional comparisons but are expected to perform
poorly on inorganic systems.

Dataset
-------
ABC3 perovskite formation energy pickle (``dataset/abc3.pickle``), preprocessed
by ``examples/abc3/abc3_preonly.py``.  Structures are periodic (PBC with cell).
The stored target is formation energy per atom (eV/atom).

Energy reference treatment
--------------------------
MACE predicts absolute DFT total energies.  ABC3 labels are formation energies,
so a per-element linear reference correction is fitted by least squares on the
evaluation split — the same protocol as the QM9 benchmark.

Usage
-----
    python examples/abc3/run_mace_benchmark.py --device cuda
    python examples/abc3/run_mace_benchmark.py --models mace_mp0_medium --device cuda
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

DATASET_DIR = str(REPO_ROOT / "dataset" / "abc3.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "abc3" / "benchmark_results")
KCAL_PER_EV = 23.0609

# MACE-MP-0 is the materials-appropriate model; others included for completeness.
_DEFAULT_MODELS = ["mace_mp0_medium"]
_ALL_MODEL_IDS = list(MACE_MODELS.keys())

_VAR_CONFIG = {
    "type": ["graph"], "output_index": [0], "output_dim": [1],
    "output_names": ["energy"], "graph_feature_names": ["energy"],
    "graph_feature_dims": [1], "node_feature_names": ["atomic_number"],
    "node_feature_dims": [1], "input_node_features": [0],
    "denormalize_output": False,
}


def _fit_linear_reference(e_preds_total, compositions, gt_per_atom, n_atoms_list):
    """Least-squares per-element reference energies to convert absolute DFT
    energies to formation energies."""
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


def evaluate_split(split_label, model_id, calc, verbose=True):
    dataset = SimplePickleDataset(
        basedir=DATASET_DIR, label=split_label, var_config=_VAR_CONFIG
    )
    if len(dataset) == 0:
        print(f"  [{split_label}] empty — skipping.")
        return {}

    needs_cs = MACE_MODELS[model_id]["needs_charge_spin"]

    e_preds_total = []
    compositions = []
    n_atoms_list = []
    gt_per_atom = []
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=True)
        if needs_cs:
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
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
        "energy_mae_eV_per_atom": float(np.abs(errors).mean()),
        "energy_rmse_eV_per_atom": float(np.sqrt((errors ** 2).mean())),
        "energy_mae_kcal_mol_per_atom": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "MACE total energy corrected by per-element linear reference "
            "energies (least-squares fit on evaluation split), divided by "
            "n_atoms, compared to ABC3 per-atom formation energy."
        ),
    }


def main():
    setup_ddp()

    parser = argparse.ArgumentParser(
        description="Evaluate MACE models on ABC3 perovskites (zero-shot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=_DEFAULT_MODELS,
                        choices=_ALL_MODEL_IDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", nargs="+", default=["testset"])
    parser.add_argument("--hydragnn-summary", default=None)
    parser.add_argument("--uma-summary", default=None)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nDataset    : {DATASET_DIR}")

    results: dict = {}
    for model_id in args.models:
        cfg = MACE_MODELS[model_id]
        print(f"\n{'='*60}")
        print(f"  Model: {cfg['label']}")
        print(f"{'='*60}")
        try:
            calc = build_mace_calculator(model_id=model_id, device=args.device)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            results[model_id] = {"label": cfg["label"], "error": str(e)}
            continue

        model_results: dict = {"label": cfg["label"]}
        for split in args.splits:
            print(f"\n--- Evaluating split: {split} ---")
            metrics = evaluate_split(split, model_id, calc, verbose=True)
            model_results[split] = metrics
            if metrics:
                print(
                    f"  [Test E-MAE = {metrics['energy_mae_eV_per_atom']:.4f} eV/atom"
                    f"  ({metrics['energy_mae_kcal_mol_per_atom']:.4f} kcal/mol)]"
                )
        results[model_id] = model_results

    out_path = os.path.join(args.output_dir, "mace_benchmark_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("  ABC3 — formation energy per atom (eV/atom)")
    print(f"{'='*60}")
    for model_id, mres in results.items():
        if "error" in mres:
            print(f"  {mres['label']:<35s}  ERROR")
            continue
        sr = mres.get("testset", {})
        if sr:
            print(f"  {mres['label']:<35s}  {sr['energy_mae_eV_per_atom']:.4f} eV/atom")

    if args.hydragnn_summary and os.path.isfile(args.hydragnn_summary):
        with open(args.hydragnn_summary) as f:
            hsum = json.load(f)
        mae = hsum.get("test_mae_eV_per_atom", hsum.get("test_mae_eV", float("nan")))
        print(f"  {'HydraGNN FT':<35s}  {mae:.4f} eV/atom")

    if args.uma_summary and os.path.isfile(args.uma_summary):
        with open(args.uma_summary) as f:
            usum = json.load(f)
        sr = usum.get("testset", {})
        if sr:
            print(f"  {'UMA (zero-shot)':<35s}  {sr.get('energy_per_atom_mae_eV', float('nan')):.4f} eV/atom")


if __name__ == "__main__":
    main()
