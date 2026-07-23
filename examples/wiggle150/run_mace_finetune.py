#!/usr/bin/env python3
"""MACE fine-tuning on Wiggle150 (relative conformational energies, energy-only).

Fine-tunes each of the 3 MACE foundation models on the Wiggle150 train/val
splits, evaluates on the test split with the min-shift-per-composition-group
convention (matching the zero-shot benchmark), and saves results to
benchmark_results/mace_finetuned_summary.json.

Wiggle150 provides only relative conformational energies (no forces), so we
train energy-only (forces_weight = 0) and let mace_run_train fit per-element
E0s automatically.

Usage
-----
    python examples/wiggle150/run_mace_finetune.py
    python examples/wiggle150/run_mace_finetune.py --models mace_off_medium
    python examples/wiggle150/run_mace_finetune.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))

from utils.finetune_utils import (
    atoms_list_to_xyz,
    pickle_split_to_atoms_list,
)
from utils.mace_calculator import MACE_MODELS

KCAL_PER_EV = 23.0609
DATASET_PATH = str(REPO_ROOT / "dataset" / "wiggle150.pickle")
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
FINETUNE_MODEL_DIR = REPO_ROOT / "pretrained_model_ensemble" / "mace_finetuned" / "wiggle150"
MACE_RUN_TRAIN = str(Path(sys.executable).parent / "mace_run_train")

_CACHE = REPO_ROOT / "mace_cache" / "mace"
CHECKPOINT_PATHS: dict[str, str] = {
    "mace_off_medium": str(_CACHE / "MACE-OFF23_medium.model"),
    "mace_polar_m":    str(_CACHE / "MACEPOLAR1Mmodel"),
    "mace_mh1_omol":   str(_CACHE / "macemh1model"),
}


def _composition_key(data) -> tuple:
    z = data.x.detach().cpu().view(-1).long().tolist()
    counts: dict[int, int] = {}
    for zi in z:
        counts[zi] = counts.get(zi, 0) + 1
    return tuple(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Fine-tuning (energy-only)
# ---------------------------------------------------------------------------

def finetune_mace(
    model_id: str,
    train_xyz: str,
    val_xyz: str,
    n_epochs: int,
    device: str,
    work_dir: Path,
    lr: float = 1e-3,
) -> tuple[Path, float]:
    model_dir = work_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_id}_wiggle150"

    cmd = [
        MACE_RUN_TRAIN,
        "--name",              run_name,
        "--foundation_model",  CHECKPOINT_PATHS[model_id],
        "--train_file",        str(train_xyz),
        "--valid_file",        str(val_xyz),
        "--energy_key",        "REF_energy",
        "--forces_key",        "REF_forces",
        "--loss",              "weighted",
        "--energy_weight",     "1.0",
        "--forces_weight",     "0.0",
        "--E0s",               "average",
        "--lr",                str(lr),
        "--max_num_epochs",    str(n_epochs),
        "--batch_size",        "4",
        "--valid_batch_size",  "4",
        "--device",            device,
        "--default_dtype",     "float64",
        "--work_dir",          str(work_dir),
        "--model_dir",         str(model_dir),
        "--checkpoints_dir",   str(work_dir / "checkpoints"),
        "--results_dir",       str(work_dir / "results"),
        "--compute_stress",    "False",
        "--log_level",         "WARNING",
    ]

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    training_sec = time.perf_counter() - t0

    candidates = sorted(model_dir.glob("*.model"))
    if not candidates:
        raise RuntimeError(
            f"mace_run_train did not produce a .model file in {model_dir}"
        )
    return candidates[0], training_sec


# ---------------------------------------------------------------------------
# Evaluation (min-shift per composition group, same as zero-shot benchmark)
# ---------------------------------------------------------------------------

def evaluate_split(
    model_path: Path,
    split: str,
    model_id: str,
    device: str,
    verbose: bool = True,
) -> dict:
    from mace.calculators import MACECalculator
    from utils.uma_calculator import pyg_data_to_ase_atoms

    sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

    _VAR_CFG = {
        "type": ["graph"], "output_index": [0], "output_dim": [1],
        "output_names": ["energy"], "graph_feature_names": ["energy"],
        "graph_feature_dims": [1], "node_feature_names": ["atomic_number"],
        "node_feature_dims": [1], "input_node_features": [0],
        "denormalize_output": False,
    }
    dataset = SimplePickleDataset(
        basedir=DATASET_PATH, label=split, var_config=_VAR_CFG
    )

    calc = MACECalculator(
        model_paths=[str(model_path)],
        device=device,
        default_dtype="float64",
    )
    cfg = MACE_MODELS[model_id]
    split_label = split.replace("set", "")

    groups: dict = defaultdict(lambda: {"mace": [], "gt": []})
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        if cfg["needs_charge_spin"]:
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
        atoms.calc = calc

        e_mace = float(atoms.get_potential_energy())
        y = data.y.detach().cpu()
        e_gt = float(y.item() if y.dim() == 0 else y.view(-1)[0])

        comp_key = _composition_key(data)
        groups[comp_key]["mace"].append(e_mace)
        groups[comp_key]["gt"].append(e_gt)

        if verbose and (i + 1) % 50 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    _inference_sec = time.perf_counter() - _t0

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
        "energy_mae_eV":       float(np.abs(errors).mean()),
        "energy_rmse_eV":      float(np.sqrt((errors ** 2).mean())),
        "energy_mae_kcal_mol": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "MACE total energies made relative within each composition group "
            "(min-shifted) to match the Wiggle150 convention."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", nargs="+", default=list(CHECKPOINT_PATHS.keys()),
        choices=list(CHECKPOINT_PATHS.keys()),
        help="MACE model IDs to fine-tune (default: all).",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3, tuned for fine-tuning).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--keep-work-dir", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset …")
    train_atoms = pickle_split_to_atoms_list(DATASET_PATH, "trainset", "conformational")
    val_atoms   = pickle_split_to_atoms_list(DATASET_PATH, "valset",   "conformational")
    print(f"  train={len(train_atoms)}, val={len(val_atoms)}")

    tmp_data_dir = Path(tempfile.mkdtemp(prefix="mace_wiggle_data_"))
    train_xyz = tmp_data_dir / "train.xyz"
    val_xyz   = tmp_data_dir / "val.xyz"

    # Energy-only dataset: attach zero forces so mace_run_train can parse the
    # forces_key column (forces_weight = 0 means they do not affect the loss).
    for atoms in train_atoms + val_atoms:
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin", 1)
        atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))

    atoms_list_to_xyz(train_atoms, str(train_xyz), has_forces=True)
    atoms_list_to_xyz(val_atoms,   str(val_xyz),   has_forces=True)
    print(f"  XYZ files written to {tmp_data_dir}")

    all_results: dict = {}

    for model_id in args.models:
        label = MACE_MODELS[model_id]["label"]
        print(f"\n{'='*60}\nFine-tuning {label} on Wiggle150 …\n{'='*60}")

        work_dir = Path(tempfile.mkdtemp(prefix=f"mace_ft_{model_id}_"))
        saved_model = FINETUNE_MODEL_DIR / f"{model_id}.model"

        try:
            model_path, training_sec = finetune_mace(
                model_id=model_id,
                train_xyz=str(train_xyz),
                val_xyz=str(val_xyz),
                n_epochs=args.epochs,
                device=args.device,
                work_dir=work_dir,
                lr=args.lr,
            )
            shutil.copy(str(model_path), str(saved_model))
            print(f"  Training done in {training_sec:.1f}s → {saved_model}")
        except Exception as exc:
            print(f"  ERROR during training: {exc}")
            all_results[model_id] = {"model_name": label, "error": str(exc)}
            continue
        finally:
            if not args.keep_work_dir:
                shutil.rmtree(str(work_dir), ignore_errors=True)

        print("  Evaluating on test set …")
        test_metrics = evaluate_split(
            model_path=saved_model,
            split="testset",
            model_id=model_id,
            device=args.device,
        )

        all_results[model_id] = {
            "model_name":        label,
            "n_epochs":          args.epochs,
            "n_train":           len(train_atoms),
            "training_wall_sec": round(training_sec, 2),
            "testset":           test_metrics,
        }
        print(f"  Test E-MAE = {test_metrics['energy_mae_eV']:.4f} eV")

    out_path = RESULTS_DIR / "mace_finetuned_summary.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved to {out_path}")

    shutil.rmtree(str(tmp_data_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
