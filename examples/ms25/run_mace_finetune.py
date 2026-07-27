#!/usr/bin/env python3
"""MACE fine-tuning on MS25 periodic systems (energy + forces).

Fine-tunes a MACE foundation model on each MS25 system's train split via
``mace_run_train`` (subprocess), evaluates on the test split, and writes
``benchmark_results/mace_finetuned_summary.json`` keyed by system.

The default foundation model is **MACE-MP-0** (materials, PBE), matching the
periodic inorganic character of MS25.  Structures are written as periodic
extended-XYZ (lattice + pbc), and the training loss uses energy + forces where
forces are available, energy-only otherwise.

Usage
-----
    python examples/ms25/run_mace_finetune.py --device cuda
    python examples/ms25/run_mace_finetune.py --systems MgO-2x2 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))

from utils.uma_calculator import pyg_data_to_ase_atoms
from utils.mace_calculator import MACE_MODELS

KCAL_PER_EV = 23.0609
PICKLE_TAG = "mlip_peratom"
DATASET_ROOT = REPO_ROOT / "dataset"
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
FINETUNE_MODEL_DIR = REPO_ROOT / "pretrained_model_ensemble" / "mace_finetuned" / "ms25"
MACE_RUN_TRAIN = str(Path(sys.executable).parent / "mace_run_train")

MS25_SYSTEMS = [
    "MgO-2x2", "MgO-4x4", "H2O-64", "H2O-192",
    "CHA", "HEA", "Reaction", "Zr-O",
]

_DEFAULT_MODELS = ["mace_mp0_medium"]

# md17-style cache paths for the organic foundation checkpoints.
_CACHE = REPO_ROOT / "mace_cache" / "mace"
CHECKPOINT_PATHS: dict[str, str] = {
    "mace_off_medium": str(_CACHE / "MACE-OFF23_medium.model"),
    "mace_polar_m":    str(_CACHE / "MACEPOLAR1Mmodel"),
    "mace_mh1_omol":   str(_CACHE / "macemh1model"),
}

_VAR_CONFIG = {
    "type": ["graph"], "output_index": [0], "output_dim": [1],
    "output_names": ["graph_energy"], "graph_feature_names": ["energy"],
    "graph_feature_dims": [1], "node_feature_names": ["atomic_number"],
    "node_feature_dims": [1], "input_node_features": [0],
    "denormalize_output": False,
}


def _dataset_dir(system: str) -> Path:
    return DATASET_ROOT / f"{system}_{PICKLE_TAG}.pickle"


def resolve_foundation(model_id: str) -> str:
    """Return the ``--foundation_model`` value for mace_run_train.

    MACE-MP-0 is available to mace_run_train via the ``medium`` keyword (it
    downloads/uses the cached MP-0 checkpoint).  The organic models are passed
    as explicit cached checkpoint paths.
    """
    if model_id == "mace_mp0_medium":
        return str(_CACHE / "20231203mace128L1_epoch199model")
    return CHECKPOINT_PATHS[model_id]


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def build_split_atoms(system: str, split: str):
    """Return (atoms_list, has_forces): periodic ASE Atoms with total-energy labels."""
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

    dataset = SimplePickleDataset(
        basedir=str(_dataset_dir(system)), label=split, var_config=_VAR_CONFIG
    )
    atoms_list = []
    has_forces = True
    for data in dataset:
        atoms = pyg_data_to_ase_atoms(data, periodic=True)
        atoms.info["REF_energy"] = float(data.energy.detach().cpu().squeeze())
        f = getattr(data, "forces", None)
        if f is None:
            f = getattr(data, "force", None)
        if f is not None:
            atoms.arrays["REF_forces"] = f.detach().cpu().numpy()
        else:
            has_forces = False
        atoms_list.append(atoms)
    return atoms_list, has_forces


def write_xyz(atoms_list, path: Path):
    from ase.io import write as ase_write
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(path), atoms_list, format="extxyz")


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune_mace(model_id, train_xyz, val_xyz, has_forces, n_epochs, device,
                  work_dir: Path, lr: float,
                  lora: bool = False, lora_rank: int = 4, lora_alpha: float = 1.0) -> tuple[Path, float]:
    model_dir = work_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_id}_ms25"

    cmd = [
        MACE_RUN_TRAIN,
        "--name",             run_name,
        "--foundation_model", resolve_foundation(model_id),
        "--train_file",       str(train_xyz),
        "--valid_file",       str(val_xyz),
        "--energy_key",       "REF_energy",
        "--loss",             "ef" if has_forces else "energy",
        "--E0s",              "average",
        "--lr",               str(lr),
        "--max_num_epochs",   str(n_epochs),
        "--batch_size",       "4",
        "--valid_batch_size", "4",
        "--device",           device,
        "--default_dtype",    "float64",
        "--work_dir",         str(work_dir),
        "--model_dir",        str(model_dir),
        "--checkpoints_dir",  str(work_dir / "checkpoints"),
        "--results_dir",      str(work_dir / "results"),
        "--compute_stress",   "False",
        "--log_level",        "WARNING",
    ]
    if has_forces:
        cmd += ["--forces_key", "REF_forces"]
    if lora:
        cmd += [
            "--lora", "True",
            "--lora_rank", str(lora_rank),
            "--lora_alpha", str(lora_alpha),
            "--ema",
            "--ema_decay", "0.995",
            "--amsgrad",
            "--clip_grad", "10.0",
            "--weight_decay", "0.0",
        ]

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    training_sec = time.perf_counter() - t0

    candidates = sorted(model_dir.glob("*.model"))
    if not candidates:
        raise RuntimeError(f"mace_run_train produced no .model in {model_dir}")
    return candidates[0], training_sec


# ---------------------------------------------------------------------------
# Evaluation (mean-centred energies per system, conservative forces)
# ---------------------------------------------------------------------------

def evaluate_split(model_path: Path, system: str, split: str, device: str,
                   verbose: bool = True) -> dict:
    from mace.calculators import MACECalculator
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

    dataset = SimplePickleDataset(
        basedir=str(_dataset_dir(system)), label=split, var_config=_VAR_CONFIG
    )
    calc = MACECalculator(model_paths=[str(model_path)], device=device,
                          default_dtype="float64")

    e_preds, e_trues = [], []
    force_errors_flat = []
    n_atoms_total = 0
    have_forces = True

    _t0 = time.perf_counter()
    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=True)
        atoms.calc = calc
        e_preds.append(float(atoms.get_potential_energy()))
        e_trues.append(float(data.energy.detach().cpu().squeeze()))
        n_atoms_total += len(atoms)

        f = getattr(data, "forces", None)
        if f is None:
            f = getattr(data, "force", None)
        if have_forces and f is not None:
            f_pred = atoms.get_forces()
            force_errors_flat.append((f_pred - f.detach().cpu().numpy()).ravel())
        else:
            have_forces = False

        if verbose and (i + 1) % 50 == 0:
            print(f"    [{system}/{split}] {i + 1}/{len(dataset)} done …")

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
    }
    if force_errors_flat:
        force_flat = np.concatenate(force_errors_flat)
        result["force_mae_eV_A"] = float(np.abs(force_flat).mean())
        result["force_rmse_eV_A"] = float(np.sqrt((force_flat ** 2).mean()))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", nargs="+", default=MS25_SYSTEMS)
    p.add_argument("--models", nargs="+", default=_DEFAULT_MODELS,
                   choices=list(MACE_MODELS.keys()))
    p.add_argument("--tag", default="",
                   help="Variant tag (e.g. 'lora'). Output: mace_finetuned_<tag>_summary.json.")
    p.add_argument("--lora", action="store_true", help="Enable native MACE LoRA fine-tuning.")
    p.add_argument("--lora-r", type=int, default=4, dest="lora_rank", help="LoRA rank (default: 4).")
    p.add_argument("--lora-alpha", type=float, default=1.0, dest="lora_alpha",
                   help="LoRA alpha scaling (default: 1.0).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Training epochs (default: 10 for LoRA, 50 for naive).")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--keep-work-dir", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lr = args.lr if args.lr is not None else (0.005 if args.lora else 1e-3)
    epochs = args.epochs if args.epochs is not None else (10 if args.lora else 50)
    _tag = args.tag
    out_path = RESULTS_DIR / (f"mace_finetuned_{_tag}_summary.json" if _tag else "mace_finetuned_summary.json")

    all_results: dict = {}

    for system in args.systems:
        if not _dataset_dir(system).is_dir():
            print(f"[SKIP] {system}: dataset missing ({_dataset_dir(system)})")
            continue

        print(f"\n### Preparing data for {system}")
        train_atoms, tf = build_split_atoms(system, "trainset")
        val_atoms, vf = build_split_atoms(system, "valset")
        has_forces = tf and vf
        print(f"  train={len(train_atoms)} val={len(val_atoms)} has_forces={has_forces}")

        tmp_data_dir = Path(tempfile.mkdtemp(prefix=f"mace_ms25_{system}_data_"))
        train_xyz = tmp_data_dir / "train.xyz"
        val_xyz = tmp_data_dir / "val.xyz"
        write_xyz(train_atoms, train_xyz)
        write_xyz(val_atoms, val_xyz)

        for model_id in args.models:
            label = MACE_MODELS[model_id]["label"]
            print(f"\n{'=' * 60}\nFine-tuning {label} on {system}\n{'=' * 60}")

            work_dir = Path(tempfile.mkdtemp(prefix=f"mace_ft_{model_id}_{system}_"))
            saved_model = FINETUNE_MODEL_DIR / f"{model_id}_{system}{'_' + _tag if _tag else ''}.model"
            _skey = f"{system}::{model_id} [{_tag}]" if _tag else f"{system}::{model_id}"
            try:
                model_path, training_sec = finetune_mace(
                    model_id, str(train_xyz), str(val_xyz), has_forces,
                    n_epochs=epochs, device=args.device, work_dir=work_dir,
                    lr=lr,
                    lora=args.lora, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                )
                shutil.copy(str(model_path), str(saved_model))
                print(f"  Training done in {training_sec:.1f}s → {saved_model}")
            except Exception as exc:
                print(f"  ERROR during training: {exc}")
                all_results[_skey] = {"model_name": label, "system": system,
                                    "error": str(exc)}
                with open(out_path, "w") as fh:
                    json.dump(all_results, fh, indent=2)
                continue
            finally:
                if not args.keep_work_dir:
                    shutil.rmtree(str(work_dir), ignore_errors=True)

            test_metrics = evaluate_split(saved_model, system, "testset", args.device)
            all_results[_skey] = {
                "model_name": label,
                "system": system,
                "n_epochs": epochs,
                "n_train": len(train_atoms),
                "training_wall_sec": round(training_sec, 2),
                "testset": test_metrics,
            }
            msg = f"  [{system}] {label}: Test E-MAE = {test_metrics['energy_mae_eV']:.4f} eV"
            if "force_mae_eV_A" in test_metrics:
                msg += f"  F-MAE = {test_metrics['force_mae_eV_A']:.4f} eV/Å"
            print(msg)

            with open(out_path, "w") as fh:
                json.dump(all_results, fh, indent=2)

        shutil.rmtree(str(tmp_data_dir), ignore_errors=True)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
