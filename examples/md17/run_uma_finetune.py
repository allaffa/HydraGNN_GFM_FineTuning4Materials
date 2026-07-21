#!/usr/bin/env python3
"""UMA fine-tuning on MD17 uracil (energy + forces).

Fine-tunes the UMA foundation model (uma-s-1p2, omol head) on the MD17 MLIP
uracil train/val splits with a custom PyTorch training loop, evaluates on the
test split (mean-centred energies + forces, matching the zero-shot benchmark),
and saves results to benchmark_results/uma_finetuned_summary.json.

UMA uses a conservative force head, so ``model.train()`` is required to enable
create_graph=True for double backprop through the force autograd.

Usage
-----
    python examples/md17/run_uma_finetune.py
    python examples/md17/run_uma_finetune.py --epochs 20 --lr 1e-4 --device cpu
    python examples/md17/run_uma_finetune.py --freeze-backbone
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))

from utils.finetune_utils import pickle_split_to_atoms_list
from utils.uma_finetune import load_trainable_uma, uma_energy_forces

KCAL_PER_EV = 23.0609
DATASET_PATH = str(REPO_ROOT / "dataset" / "md17_mlip.pickle")
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
TASK_NAME = "omol"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_uma(
    model,
    calc,
    train_atoms,
    n_epochs: int,
    lr: float,
    batch_size: int,
    force_weight: float,
    freeze_backbone: bool,
    verbose: bool = True,
) -> float:
    """Custom energy+force training loop.  Returns training wall-clock seconds."""
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    model.train()
    dtype = next(model.parameters()).dtype

    # Pre-extract targets.
    targets = []
    for atoms in train_atoms:
        e = float(atoms.info["REF_energy"])
        f = torch.tensor(atoms.arrays["REF_forces"], dtype=dtype)
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
            e_loss = (e_pred.squeeze() - e_true) ** 2
            f_loss = (f_pred - f_true).pow(2).mean()
            loss = e_loss + force_weight * f_loss
            (loss / batch_size).backward()
            running += float(loss)

            if (step + 1) % batch_size == 0 or (step + 1) == n:
                opt.step()
                opt.zero_grad()

        if verbose:
            print(f"    epoch {epoch + 1}/{n_epochs}  mean_loss={running / n:.4f}")

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Evaluation (mean-centred energies + forces, same as zero-shot benchmark)
# ---------------------------------------------------------------------------

def evaluate_split(model, calc, split: str, verbose: bool = True) -> dict:
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
    from utils.uma_calculator import pyg_data_to_ase_atoms

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

    model.eval()
    split_label = split.replace("set", "")

    e_preds, e_trues = [], []
    force_errors_flat = []
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = calc

        e_preds.append(float(atoms.get_potential_energy()))
        f_pred = atoms.get_forces()

        e_trues.append(float(data.energy.detach().cpu().squeeze()))
        f_true = data.forces.detach().cpu().numpy()
        force_errors_flat.append((f_pred - f_true).ravel())

        if verbose and (i + 1) % 50 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    _inference_sec = time.perf_counter() - _t0

    e_preds = np.asarray(e_preds)
    e_trues = np.asarray(e_trues)
    e_errors = (e_preds - e_preds.mean()) - (e_trues - e_trues.mean())
    force_flat = np.concatenate(force_errors_flat)

    return {
        "n_structures": len(e_errors),
        "inference_wall_sec": round(_inference_sec, 2),
        "energy_mae_eV":        float(np.abs(e_errors).mean()),
        "energy_rmse_eV":       float(np.sqrt((e_errors ** 2).mean())),
        "energy_mae_kcal_mol":  float(np.abs(e_errors).mean()) * KCAL_PER_EV,
        "force_mae_eV_A":       float(np.abs(force_flat).mean()),
        "force_rmse_eV_A":      float(np.sqrt((force_flat ** 2).mean())),
        "force_mae_kcal_mol_A": float(np.abs(force_flat).mean()) * KCAL_PER_EV,
        "note": (
            "Energies mean-centred before MAE; forces evaluated without "
            "centring (reference-invariant)."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="uma-s-1p2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--force-weight", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Fine-tune only the output head (faster, more stable).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading dataset …")
    train_atoms = pickle_split_to_atoms_list(DATASET_PATH, "trainset", "mlip")
    print(f"  train={len(train_atoms)}")

    print(f"Loading UMA ({args.model_name}) …")
    pu, calc, model = load_trainable_uma(args.model_name, TASK_NAME, args.device)

    print("Fine-tuning …")
    training_sec = train_uma(
        model=model,
        calc=calc,
        train_atoms=train_atoms,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        force_weight=args.force_weight,
        freeze_backbone=args.freeze_backbone,
    )
    print(f"  Training done in {training_sec:.1f}s")

    print("Evaluating on test set …")
    test_metrics = evaluate_split(model, calc, "testset")

    result = {
        args.model_name: {
            "model_name":        f"UMA {args.model_name} (fine-tuned)",
            "n_epochs":          args.epochs,
            "n_train":           len(train_atoms),
            "lr":                args.lr,
            "force_weight":      args.force_weight,
            "freeze_backbone":   args.freeze_backbone,
            "training_wall_sec": round(training_sec, 2),
            "testset":           test_metrics,
        }
    }

    out_path = RESULTS_DIR / "uma_finetuned_summary.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {out_path}")
    print(
        f"  Test E-MAE = {test_metrics['energy_mae_eV']:.4f} eV  "
        f"F-MAE = {test_metrics['force_mae_eV_A']:.4f} eV/Å"
    )


if __name__ == "__main__":
    main()
