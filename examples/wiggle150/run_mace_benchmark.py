#!/usr/bin/env python3
"""Wiggle150 MACE state-of-the-art benchmark.

Evaluates three MACE foundation models (MACE-OFF23 medium, MACE-POLAR-1
polar-1-m, MACE-MH-1 omol head) on the Wiggle150 test split and compares
the results to the HydraGNN fine-tuning numbers from ``run_benchmark.py``
and optionally the UMA numbers from ``run_uma_benchmark.py``.

Dataset
-------
Wiggle150 pickle (``dataset/wiggle150.pickle``), preprocessed by
``examples/wiggle150/wiggle150_preonly.py``.  Energies are in eV (relative
conformer energies: E_conf − E_min_conf per molecule).  Non-periodic.

Energy reference treatment
--------------------------
Wiggle150 stores *relative* conformer energies while MACE predicts *total*
potential energies.  To align both scales we:

  1. Group structures by composition fingerprint (proxy for molecule identity,
     since no mol_id is stored in the pickle).
  2. Within each group subtract the minimum MACE total energy, obtaining MACE
     relative conformational energies that match the GT convention.
  3. Compute MAE between MACE relative energies and GT relative energies.

Usage
-----
    python examples/wiggle150/run_mace_benchmark.py

    # Select a subset of models
    python examples/wiggle150/run_mace_benchmark.py \\
        --models mace_off_medium mace_mh1_omol

    # Full comparison table (MACE + UMA + HydraGNN)
    python examples/wiggle150/run_mace_benchmark.py \\
        --hydragnn-summary examples/wiggle150/benchmark_results/benchmark_summary.json \\
        --uma-summary      examples/wiggle150/benchmark_results/uma_benchmark_summary.json

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
from collections import defaultdict
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
DATASET_DIR = str(REPO_ROOT / "dataset" / "wiggle150.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "wiggle150" / "benchmark_results")
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
# Helpers
# ---------------------------------------------------------------------------

def _composition_key(data) -> tuple:
    """Return a hashable (element, count) fingerprint as molecule identity proxy."""
    import torch
    z = data.x[:, 0].long()
    unique, counts = torch.unique(z, return_counts=True)
    return tuple(zip(unique.tolist(), counts.tolist()))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    split_label: str,
    model_id: str,
    device: str | None,
    verbose: bool = True,
) -> dict:
    """Evaluate a single MACE model on one Wiggle150 split.

    Groups structures by composition fingerprint and min-shifts MACE total
    energies within each group to obtain relative conformational energies
    matching the GT convention (E_conf - E_min_conf).
    """
    dataset = SimplePickleDataset(
        basedir=DATASET_DIR, label=split_label, var_config=_VAR_CONFIG
    )
    if len(dataset) == 0:
        print(f"  [{split_label}] empty — skipping.")
        return {}

    calc = build_mace_calculator(model_id=model_id, device=device)
    needs_cs = MACE_MODELS[model_id]["needs_charge_spin"]

    groups: dict = defaultdict(lambda: {"mace": [], "gt": []})
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        if needs_cs:
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
        atoms.calc = calc

        e_mace = float(atoms.get_potential_energy())

        y = data.y.detach().cpu()
        e_gt = float(y.item() if y.dim() == 0 else y.view(-1)[0])

        comp_key = _composition_key(data)
        groups[comp_key]["mace"].append(e_mace)
        groups[comp_key]["gt"].append(e_gt)

        if verbose and (i + 1) % 100 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    _inference_sec = time.perf_counter() - _t0
    # Compute relative MACE energies per composition group and collect errors
    errors = []
    for grp in groups.values():
        mace_arr = np.asarray(grp["mace"])
        gt_arr = np.asarray(grp["gt"])
        mace_rel = mace_arr - mace_arr.min()
        errors.extend((mace_rel - gt_arr).tolist())

    errors = np.asarray(errors)

    return {
        "n_structures": len(errors),
        "n_composition_groups": len(groups),
        "inference_wall_sec": round(_inference_sec, 2),
        "energy_mae_eV": float(np.abs(errors).mean()),
        "energy_rmse_eV": float(np.sqrt((errors ** 2).mean())),
        "energy_mae_kcal_mol": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "MACE total energies are min-shifted within each composition group "
            "to match the Wiggle150 relative-energy convention."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(
    mace_results: dict,
    uma_summary: dict | None,
    hydragnn_summary: dict | None,
    mace_finetuned: dict | None = None,
    uma_finetuned: dict | None = None,
):
    KCAL = KCAL_PER_EV
    hdr = f"\n{'Model':<30s}  {'E-MAE (eV)':>10s}  {'E-MAE (kcal/mol)':>17s}"
    sep = "-" * 62

    print("\n" + "=" * 70)
    print("  Wiggle150 — MACE / UMA / HydraGNN comparison")
    print("=" * 70)
    print(hdr)
    print(sep)

    timing_entries: list[tuple] = []

    def _row(label, e_mae):
        print(f"{label:<30s}  {e_mae:10.4f}  {e_mae * KCAL:17.4f}")

    # MACE rows (zero-shot)
    for mid in _ALL_MODEL_IDS:
        if mid not in mace_results:
            continue
        sr = mace_results[mid].get("testset", {})
        if sr:
            _row(MACE_MODELS[mid]["label"], sr["energy_mae_eV"])
            timing_entries.append(
                (MACE_MODELS[mid]["label"], None, sr.get("inference_wall_sec"))
            )

    # MACE rows (fine-tuned)
    if mace_finetuned:
        for mid in _ALL_MODEL_IDS:
            if mid not in mace_finetuned:
                continue
            fr = mace_finetuned[mid]
            sr = fr.get("testset", {})
            if sr:
                label = f"{MACE_MODELS[mid]['label']} (FT)"
                _row(label, sr["energy_mae_eV"])
                timing_entries.append(
                    (label, fr.get("training_wall_sec"), sr.get("inference_wall_sec"))
                )

    # UMA row
    if uma_summary:
        ur = uma_summary.get("testset", {})
        if ur:
            model_name = uma_summary.get("model_name", "uma-s-1p2")
            _row(f"UMA {model_name}", ur.get("energy_mae_eV", float("nan")))
            timing_entries.append(
                (f"UMA {model_name}", None, ur.get("inference_wall_sec"))
            )

    # UMA rows (fine-tuned)
    if uma_finetuned:
        for name, fr in uma_finetuned.items():
            sr = fr.get("testset", {})
            if sr:
                label = f"UMA {name} (FT)"
                _row(label, sr.get("energy_mae_eV", float("nan")))
                timing_entries.append(
                    (label, fr.get("training_wall_sec"), sr.get("inference_wall_sec"))
                )

    # HydraGNN rows
    if hydragnn_summary:
        labels = {
            "frozen_fp64": "HydraGNN (frozen, fp64)",
            "unfrozen_fp64": "HydraGNN (unfrozen, fp64)",
            "scratch_fp64": "HydraGNN (scratch, fp64)",
            "frozen": "HydraGNN (frozen FT)",
            "unfrozen": "HydraGNN (unfrozen FT)",
            "scratch": "HydraGNN (scratch)",
        }
        printed = set()
        for key, label in labels.items():
            if key not in hydragnn_summary or key in printed:
                continue
            s = hydragnn_summary[key]
            e_mae = s.get(
                "best_val_mae_eV",
                s.get("best_mae_eV", s.get("best_energy_mae_eV", float("nan"))),
            )
            _row(label, e_mae)
            timing_entries.append(
                (label, s.get("training_wall_sec"), s.get("inference_wall_sec"))
            )
            printed.add(key)

    print()
    print_timing_summary(timing_entries)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_ddp()

    parser = argparse.ArgumentParser(
        description="Evaluate MACE models on Wiggle150 and compare to HydraGNN / UMA.",
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

    # Auto-load fine-tuned summaries from the output directory if present.
    mace_finetuned = None
    ft_mace_path = os.path.join(args.output_dir, "mace_finetuned_summary.json")
    if os.path.isfile(ft_mace_path):
        with open(ft_mace_path) as f:
            mace_finetuned = json.load(f)

    # Merge every UMA fine-tune summary (e.g. tagged full-FT + frozen variants)
    # so each appears as its own row in the comparison table.
    uma_finetuned = None
    import glob as _glob
    _uma_ft_files = sorted(
        _glob.glob(os.path.join(args.output_dir, "uma_finetuned*summary.json"))
    )
    if _uma_ft_files:
        uma_finetuned = {}
        for _p in _uma_ft_files:
            with open(_p) as f:
                uma_finetuned.update(json.load(f))

    print_comparison_table(
        results, uma_summary, hydragnn_summary, mace_finetuned, uma_finetuned
    )


if __name__ == "__main__":
    main()
