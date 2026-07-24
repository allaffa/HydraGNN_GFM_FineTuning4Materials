#!/usr/bin/env python3
"""UMA fine-tuning on MS25 periodic systems (energy + forces).

Fine-tunes UMA (uma-s-1p2, ``omat`` task) on each MS25 system's train split
with a custom PyTorch loop, evaluates on the test split (energies mean-centred
per system, conservative forces), and writes
``benchmark_results/uma_finetuned_summary.json`` keyed by system.

UMA uses a conservative force head, so ``model.train()`` is required to enable
create_graph=True for double backprop through the force autograd.

Usage
-----
    python examples/ms25/run_uma_finetune.py --device cuda
    python examples/ms25/run_uma_finetune.py --systems MgO-2x2 --epochs 20
    python examples/ms25/run_uma_finetune.py --freeze-backbone
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))

from utils.uma_calculator import pyg_data_to_ase_atoms
from utils.uma_finetune import load_trainable_uma, uma_energy_forces

KCAL_PER_EV = 23.0609
PICKLE_TAG = "mlip_peratom"
DATASET_ROOT = REPO_ROOT / "dataset"
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
TASK_NAME = "omat"

MS25_SYSTEMS = [
    "MgO-2x2", "MgO-4x4", "H2O-64", "H2O-192",
    "CHA", "HEA", "Reaction", "Zr-O",
]

_VAR_CONFIG = {
    "type": ["graph"], "output_index": [0], "output_dim": [1],
    "output_names": ["graph_energy"], "graph_feature_names": ["energy"],
    "graph_feature_dims": [1], "node_feature_names": ["atomic_number"],
    "node_feature_dims": [1], "input_node_features": [0],
    "denormalize_output": False,
}


def _dataset_dir(system: str) -> Path:
    return DATASET_ROOT / f"{system}_{PICKLE_TAG}.pickle"


def load_system_atoms(system: str, split: str):
    """Return (atoms_list, has_forces) with periodic cells and total-energy labels."""
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

    basedir = str(_dataset_dir(system))
    dataset = SimplePickleDataset(basedir=basedir, label=split, var_config=_VAR_CONFIG)

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_uma(model, calc, train_atoms, has_forces, n_epochs, lr, batch_size,
              force_weight, freeze_backbone, lora=False, lora_r=8, lora_alpha=16.0,
              verbose=True) -> float:
    if lora:
        from utils.uma_finetune import apply_lora_to_backbone
        n_lora = apply_lora_to_backbone(model.backbone, r=lora_r, alpha=lora_alpha)
        if verbose:
            print(f"    [LoRA] {n_lora} adapters injected (r={lora_r}, α={lora_alpha})")
        for name, param in model.backbone.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad_(False)
    elif freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    model.train()
    dtype = next(model.parameters()).dtype

    targets = []
    for atoms in train_atoms:
        e = float(atoms.info["REF_energy"])
        f = (torch.tensor(atoms.arrays["REF_forces"], dtype=dtype)
             if has_forces else None)
        targets.append((e, f))

    t0 = time.perf_counter()
    n = len(train_atoms)
    for epoch in range(n_epochs):
        order = np.random.permutation(n)
        opt.zero_grad()
        running = 0.0
        for step, idx in enumerate(order):
            atoms = train_atoms[int(idx)]
            e_true, f_true = targets[int(idx)]

            e_pred, f_pred = uma_energy_forces(model, calc, atoms, TASK_NAME)
            loss = (e_pred.squeeze() - e_true) ** 2
            if has_forces and f_true is not None:
                loss = loss + force_weight * (f_pred - f_true).pow(2).mean()
            (loss / batch_size).backward()
            running += float(loss)

            if (step + 1) % batch_size == 0 or (step + 1) == n:
                opt.step()
                opt.zero_grad()

        if verbose:
            print(f"    epoch {epoch + 1}/{n_epochs}  mean_loss={running / n:.4f}")

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Evaluation (mean-centred energies, conservative forces)
# ---------------------------------------------------------------------------

def evaluate_split(model, calc, system, split, verbose=True) -> dict:
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

    basedir = str(_dataset_dir(system))
    dataset = SimplePickleDataset(basedir=basedir, label=split, var_config=_VAR_CONFIG)

    model.eval()
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
    p.add_argument("--model-name", default="uma-s-1p2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--force-weight", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--tag", default=None,
                   help="Variant tag; writes uma_finetuned_{tag}_summary.json and suffixes result keys.")
    p.add_argument("--lora", action="store_true",
                   help="LoRA fine-tuning of backbone scalar linear layers (preserves equivariance).")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8).")
    p.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling (default: 16.0).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _summary_name = (
        f"uma_finetuned_{args.tag}_summary.json" if args.tag
        else "uma_finetuned_summary.json"
    )
    out_path = RESULTS_DIR / _summary_name

    all_results: dict = {}
    for system in args.systems:
        _skey = f"{system} [{args.tag}]" if args.tag else system
        basedir = _dataset_dir(system)
        if not basedir.is_dir():
            print(f"[SKIP] {system}: dataset missing ({basedir})")
            continue

        print(f"\n{'=' * 60}\n  Fine-tuning UMA on {system}\n{'=' * 60}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        train_atoms, has_forces = load_system_atoms(system, "trainset")
        print(f"  train={len(train_atoms)}  has_forces={has_forces}")

        # Reload a fresh UMA per system so systems do not leak weights.
        pu, calc, model = load_trainable_uma(args.model_name, TASK_NAME, args.device)

        try:
            training_sec = train_uma(
                model, calc, train_atoms, has_forces,
                n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                force_weight=args.force_weight, freeze_backbone=args.freeze_backbone,
                lora=args.lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            )
        except Exception as exc:
            print(f"  ERROR during training: {exc}")
            all_results[_skey] = {"model_name": f"UMA {args.model_name} (FT)",
                                   "error": str(exc)}
            with open(out_path, "w") as fh:
                json.dump(all_results, fh, indent=2)
            continue

        print(f"  Training done in {training_sec:.1f}s — evaluating …")
        test_metrics = evaluate_split(model, calc, system, "testset")

        all_results[_skey] = {
            "model_name": f"UMA {args.model_name} (FT)",
            "n_epochs": args.epochs,
            "n_train": len(train_atoms),
            "lr": args.lr,
            "force_weight": args.force_weight,
            "freeze_backbone": args.freeze_backbone,
            "lora": args.lora,
            "training_wall_sec": round(training_sec, 2),
            "testset": test_metrics,
        }
        msg = f"  [{system}] Test E-MAE = {test_metrics['energy_mae_eV']:.4f} eV"
        if "force_mae_eV_A" in test_metrics:
            msg += f"  F-MAE = {test_metrics['force_mae_eV_A']:.4f} eV/Å"
        print(msg)

        with open(out_path, "w") as fh:
            json.dump(all_results, fh, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
