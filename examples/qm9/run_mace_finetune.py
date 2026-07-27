#!/usr/bin/env python3
"""MACE fine-tuning on QM9 (per-atom atomization energy, energy-only).

Fine-tunes each of the 3 MACE foundation models on a random subset of the
QM9 train/val splits (energy-only; QM9 has no forces), evaluates on the test
split with the per-element linear reference correction (matching the zero-shot
benchmark), and saves results to benchmark_results/mace_finetuned_summary.json.

The full QM9 train split (~91k structures) is too slow to fine-tune on CPU,
so a deterministic random subset (default 5000, seed 42) is used.

Usage
-----
    python examples/qm9/run_mace_finetune.py
    python examples/qm9/run_mace_finetune.py --n-train 5000 --epochs 50
    python examples/qm9/run_mace_finetune.py --device cuda
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

from utils.finetune_utils import (
    atoms_list_to_xyz,
    pickle_split_to_atoms_list,
)
from utils.mace_calculator import MACE_MODELS

KCAL_PER_EV = 23.0609
DATASET_PATH = str(REPO_ROOT / "dataset" / "qm9_energy.pickle")
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
FINETUNE_MODEL_DIR = REPO_ROOT / "pretrained_model_ensemble" / "mace_finetuned" / "qm9"
MACE_RUN_TRAIN = str(Path(sys.executable).parent / "mace_run_train")

_CACHE = REPO_ROOT / "mace_cache" / "mace"
CHECKPOINT_PATHS: dict[str, str] = {
    "mace_off_medium": str(_CACHE / "MACE-OFF23_medium.model"),
    "mace_polar_m":    str(_CACHE / "MACEPOLAR1Mmodel"),
    "mace_mh1_omol":   str(_CACHE / "macemh1model"),
}


def _fit_linear_reference(e_preds_total, compositions, gt_per_atom, n_atoms_list):
    """Fit per-element reference energies by least squares (see zero-shot script)."""
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
    lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
) -> tuple[Path, float]:
    model_dir = work_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_id}_qm9"

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
        "--batch_size",        "16",
        "--valid_batch_size",  "16",
        "--device",            device,
        "--default_dtype",     "float64",
        "--work_dir",          str(work_dir),
        "--model_dir",         str(model_dir),
        "--checkpoints_dir",   str(work_dir / "checkpoints"),
        "--results_dir",       str(work_dir / "results"),
        "--compute_stress",    "False",
        "--log_level",         "WARNING",
    ]
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
        raise RuntimeError(
            f"mace_run_train did not produce a .model file in {model_dir}"
        )
    return candidates[0], training_sec


# ---------------------------------------------------------------------------
# Evaluation (per-element linear reference correction, same as zero-shot)
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

    e_preds_total = []
    compositions = []
    n_atoms_list = []
    gt_per_atom = []
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        if cfg["needs_charge_spin"]:
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
        gt_val = float(y.squeeze()) if y.numel() == 1 else float(y.view(-1)[0])
        gt_per_atom.append(gt_val)

        if verbose and (i + 1) % 500 == 0:
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
        "energy_per_atom_mae_eV":       float(np.abs(errors).mean()),
        "energy_per_atom_rmse_eV":      float(np.sqrt((errors ** 2).mean())),
        "energy_per_atom_mae_kcal_mol": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "MACE total energy corrected by per-element linear reference "
            "energies (least-squares fit on evaluation split), divided by "
            "n_atoms, compared to QM9 per-atom atomization energy."
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
    p.add_argument("--tag", default="",
                   help="Variant tag (e.g. 'lora'). Output: mace_finetuned_<tag>_summary.json.")
    p.add_argument("--lora", action="store_true", help="Enable native MACE LoRA fine-tuning.")
    p.add_argument("--lora-r", type=int, default=4, dest="lora_rank", help="LoRA rank (default: 4).")
    p.add_argument("--lora-alpha", type=float, default=1.0, dest="lora_alpha",
                   help="LoRA alpha scaling (default: 1.0).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Training epochs (default: 10 for LoRA, 50 for naive).")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 0.005 for LoRA, 1e-3 for naive).")
    p.add_argument("--n-train", type=int, default=5000,
                   help="Random training subset size (default: 5000, seed 42).")
    p.add_argument("--n-val", type=int, default=1000,
                   help="Random validation subset size (default: 1000).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-work-dir", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset (subsampled) …")
    train_atoms = pickle_split_to_atoms_list(
        DATASET_PATH, "trainset", "atomization",
        max_structures=args.n_train, seed=args.seed,
    )
    val_atoms = pickle_split_to_atoms_list(
        DATASET_PATH, "valset", "atomization",
        max_structures=args.n_val, seed=args.seed,
    )
    print(f"  train={len(train_atoms)}, val={len(val_atoms)}")

    tmp_data_dir = Path(tempfile.mkdtemp(prefix="mace_qm9_data_"))
    train_xyz = tmp_data_dir / "train.xyz"
    val_xyz   = tmp_data_dir / "val.xyz"

    # Energy-only: attach zero forces (forces_weight = 0 in the loss).
    for atoms in train_atoms + val_atoms:
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin", 1)
        atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))

    atoms_list_to_xyz(train_atoms, str(train_xyz), has_forces=True)
    atoms_list_to_xyz(val_atoms,   str(val_xyz),   has_forces=True)
    print(f"  XYZ files written to {tmp_data_dir}")

    lr = args.lr if args.lr is not None else (0.005 if args.lora else 1e-3)
    epochs = args.epochs if args.epochs is not None else (10 if args.lora else 50)
    _tag = args.tag
    out_path = RESULTS_DIR / (f"mace_finetuned_{_tag}_summary.json" if _tag else "mace_finetuned_summary.json")
    all_results: dict = {}

    for model_id in args.models:
        label = MACE_MODELS[model_id]["label"]
        print(f"\n{'='*60}\nFine-tuning {label} on QM9 (n={len(train_atoms)}) …\n{'='*60}")

        work_dir = Path(tempfile.mkdtemp(prefix=f"mace_ft_{model_id}_"))
        saved_model = FINETUNE_MODEL_DIR / f"{model_id}{'_' + _tag if _tag else ''}.model"
        _rkey = f"{model_id} [{_tag}]" if _tag else model_id

        try:
            model_path, training_sec = finetune_mace(
                model_id=model_id,
                train_xyz=str(train_xyz),
                val_xyz=str(val_xyz),
                n_epochs=epochs,
                device=args.device,
                work_dir=work_dir,
                lr=lr,
                lora=args.lora,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
            )
            shutil.copy(str(model_path), str(saved_model))
            print(f"  Training done in {training_sec:.1f}s → {saved_model}")
        except Exception as exc:
            print(f"  ERROR during training: {exc}")
            all_results[_rkey] = {"model_name": label, "error": str(exc)}
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

        all_results[_rkey] = {
            "model_name":        label,
            "n_epochs":          epochs,
            "n_train":           len(train_atoms),
            "training_wall_sec": round(training_sec, 2),
            "testset":           test_metrics,
        }
        print(
            f"  Test E/atom-MAE = {test_metrics['energy_per_atom_mae_eV']:.4f} eV/atom"
        )

    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved to {out_path}")

    shutil.rmtree(str(tmp_data_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
