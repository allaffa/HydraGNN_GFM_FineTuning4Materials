#!/usr/bin/env python3
"""MACE fine-tuning on ABC3 (formation energy per atom, periodic crystals).

Fine-tunes a MACE foundation model (default **MACE-MP-0**, materials/PBE) on the
ABC3 train split via ``mace_run_train``, evaluates on the test split, and writes
``benchmark_results/mace_finetuned_summary.json``.

ABC3 targets formation energy per atom (eV/atom).  MACE learns an *extensive*
energy, so the training label is the total formation energy (per-atom ×
n_atoms) written into periodic extended-XYZ (lattice + pbc).  There are no
reference forces, so the loss is energy-only.  The per-atom MAE is reported.

NOTE: requires ``data.cell`` on each Data object for a periodic evaluation;
``abc3_getData_API.py`` now persists the lattice — rebuild ``dataset/abc3.pickle``
if it predates that change.

Usage
-----
    python examples/abc3/run_mace_finetune.py --device cuda
    python examples/abc3/run_mace_finetune.py --epochs 50 --models mace_mp0_medium
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

DATASET_PATH = str(REPO_ROOT / "dataset" / "abc3.pickle")
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
FINETUNE_MODEL_DIR = REPO_ROOT / "pretrained_model_ensemble" / "mace_finetuned" / "abc3"
MACE_RUN_TRAIN = str(Path(sys.executable).parent / "mace_run_train")

_DEFAULT_MODELS = ["mace_mp0_medium"]
_CACHE = REPO_ROOT / "mace_cache" / "mace"
CHECKPOINT_PATHS: dict[str, str] = {
    "mace_off_medium": str(_CACHE / "MACE-OFF23_medium.model"),
    "mace_polar_m":    str(_CACHE / "MACEPOLAR1Mmodel"),
    "mace_mh1_omol":   str(_CACHE / "macemh1model"),
}

_VAR_CONFIG = {
    "type": ["graph"], "output_index": [0], "output_dim": [1],
    "output_names": ["energy"], "graph_feature_names": ["energy"],
    "graph_feature_dims": [1], "node_feature_names": ["atomic_number"],
    "node_feature_dims": [1], "input_node_features": [0],
    "denormalize_output": False,
}


def resolve_foundation(model_id: str) -> str:
    if model_id == "mace_mp0_medium":
        return str(_CACHE / "20231203mace128L1_epoch199model")
    return CHECKPOINT_PATHS[model_id]


def build_split_atoms(split: str):
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

    dataset = SimplePickleDataset(basedir=DATASET_PATH, label=split, var_config=_VAR_CONFIG)
    atoms_list = []
    for data in dataset:
        atoms = pyg_data_to_ase_atoms(data, periodic=True)
        e_per_atom = float(data.energy.detach().cpu().squeeze())
        atoms.info["REF_energy"] = e_per_atom * len(atoms)   # total (extensive)
        atoms_list.append(atoms)
    return atoms_list


def write_xyz(atoms_list, path: Path):
    from ase.io import write as ase_write
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(path), atoms_list, format="extxyz")


def finetune_mace(model_id, train_xyz, val_xyz, n_epochs, device,
                  work_dir: Path, lr: float,
                  lora: bool = False, lora_rank: int = 4, lora_alpha: float = 1.0) -> tuple[Path, float]:
    model_dir = work_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        MACE_RUN_TRAIN,
        "--name",             f"{model_id}_abc3",
        "--foundation_model", resolve_foundation(model_id),
        "--train_file",       str(train_xyz),
        "--valid_file",       str(val_xyz),
        "--energy_key",       "REF_energy",
        "--loss",             "energy",
        "--E0s",              "average",
        "--lr",               str(lr),
        "--max_num_epochs",   str(n_epochs),
        "--batch_size",       "8",
        "--valid_batch_size", "8",
        "--device",           device,
        "--default_dtype",    "float64",
        "--work_dir",         str(work_dir),
        "--model_dir",        str(model_dir),
        "--checkpoints_dir",  str(work_dir / "checkpoints"),
        "--results_dir",      str(work_dir / "results"),
        "--compute_stress",   "False",
        "--log_level",        "WARNING",
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
        raise RuntimeError(f"mace_run_train produced no .model in {model_dir}")
    return candidates[0], training_sec


def evaluate_split(model_path: Path, split: str, device: str, verbose=True) -> dict:
    from mace.calculators import MACECalculator
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

    dataset = SimplePickleDataset(basedir=DATASET_PATH, label=split, var_config=_VAR_CONFIG)
    calc = MACECalculator(model_paths=[str(model_path)], device=device,
                          default_dtype="float64")
    per_atom_errors = []
    _t0 = time.perf_counter()
    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=True)
        atoms.calc = calc
        e_pred_per_atom = float(atoms.get_potential_energy()) / len(atoms)
        e_true_per_atom = float(data.energy.detach().cpu().squeeze())
        per_atom_errors.append(e_pred_per_atom - e_true_per_atom)
        if verbose and (i + 1) % 50 == 0:
            print(f"    [{split}] {i + 1}/{len(dataset)} done …")
    _inference_sec = time.perf_counter() - _t0
    err = np.asarray(per_atom_errors)
    return {
        "n_structures": len(err),
        "inference_wall_sec": round(_inference_sec, 2),
        "energy_mae_eV_per_atom": float(np.abs(err).mean()),
        "energy_rmse_eV_per_atom": float(np.sqrt((err ** 2).mean())),
        "note": "Formation energy per atom; direct MAE (no centering) after fine-tuning.",
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
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

    print("Preparing ABC3 XYZ files …")
    train_atoms = build_split_atoms("trainset")
    val_atoms = build_split_atoms("valset")
    print(f"  train={len(train_atoms)} val={len(val_atoms)}")

    tmp_data_dir = Path(tempfile.mkdtemp(prefix="mace_abc3_data_"))
    train_xyz = tmp_data_dir / "train.xyz"
    val_xyz = tmp_data_dir / "val.xyz"
    write_xyz(train_atoms, train_xyz)
    write_xyz(val_atoms, val_xyz)

    all_results: dict = {}
    for model_id in args.models:
        label = MACE_MODELS[model_id]["label"]
        print(f"\n{'=' * 60}\nFine-tuning {label} on ABC3\n{'=' * 60}")
        work_dir = Path(tempfile.mkdtemp(prefix=f"mace_ft_{model_id}_abc3_"))
        saved_model = FINETUNE_MODEL_DIR / f"{model_id}{'_' + _tag if _tag else ''}.model"
        _rkey = f"{model_id} [{_tag}]" if _tag else model_id
        try:
            model_path, training_sec = finetune_mace(
                model_id, str(train_xyz), str(val_xyz),
                n_epochs=epochs, device=args.device, work_dir=work_dir, lr=lr,
                lora=args.lora, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            )
            shutil.copy(str(model_path), str(saved_model))
            print(f"  Training done in {training_sec:.1f}s → {saved_model}")
        except Exception as exc:
            print(f"  ERROR during training: {exc}")
            all_results[_rkey] = {"model_name": label, "error": str(exc)}
            with open(out_path, "w") as fh:
                json.dump(all_results, fh, indent=2)
            continue
        finally:
            if not args.keep_work_dir:
                shutil.rmtree(str(work_dir), ignore_errors=True)

        test_metrics = evaluate_split(saved_model, "testset", args.device)
        all_results[_rkey] = {
            "model_name": label,
            "n_epochs": epochs,
            "n_train": len(train_atoms),
            "training_wall_sec": round(training_sec, 2),
            "testset": test_metrics,
        }
        print(f"  Test E-MAE = {test_metrics['energy_mae_eV_per_atom']:.4f} eV/atom")
        with open(out_path, "w") as fh:
            json.dump(all_results, fh, indent=2)

    shutil.rmtree(str(tmp_data_dir), ignore_errors=True)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
