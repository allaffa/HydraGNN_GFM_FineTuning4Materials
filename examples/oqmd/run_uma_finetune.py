#!/usr/bin/env python3
"""UMA fine-tuning on OQMD (formation energy per atom, periodic crystals).

OQMD targets the DFT **formation energy per atom** (eV/atom).  Foundation MLIPs
predict an *extensive* energy, so we fine-tune UMA (uma-s-1p2, ``omat`` task) to
regress the total formation energy (= per-atom × n_atoms) and report the
per-atom MAE on the held-out test split.

There are no reference forces in OQMD, so training is energy-only.  Because
compositions vary across samples, energies are **not** mean-centred; after
fine-tuning the model predicts the absolute (element-referenced) target.

NOTE: a physically-correct periodic evaluation requires ``data.cell`` on each
Data object.  ``oqmd_getData.py`` now persists the lattice; rebuild the
``dataset/oqmd.pickle`` dataset if it predates that change.

Usage
-----
    python examples/oqmd/run_uma_finetune.py --device cuda
    python examples/oqmd/run_uma_finetune.py --epochs 30 --freeze-backbone
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

from utils.uma_calculator import pyg_data_to_ase_atoms
from utils.uma_finetune import load_trainable_uma, uma_energy_forces

DATASET_PATH = str(REPO_ROOT / "dataset" / "oqmd.pickle")
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
TASK_NAME = "omat"

_VAR_CONFIG = {
    "type": ["graph"], "output_index": [0], "output_dim": [1],
    "output_names": ["energy"], "graph_feature_names": ["energy"],
    "graph_feature_dims": [1], "node_feature_names": ["atomic_number"],
    "node_feature_dims": [1], "input_node_features": [0],
    "denormalize_output": False,
}


def load_atoms(split: str):
    """Return periodic ASE Atoms with total formation energy in info['REF_energy']."""
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

    dataset = SimplePickleDataset(basedir=DATASET_PATH, label=split, var_config=_VAR_CONFIG)
    atoms_list = []
    for data in dataset:
        atoms = pyg_data_to_ase_atoms(data, periodic=True)
        e_per_atom = float(data.energy.detach().cpu().squeeze())
        atoms.info["REF_energy"] = e_per_atom * len(atoms)   # total (extensive)
        atoms.info["REF_energy_per_atom"] = e_per_atom
        atoms_list.append(atoms)
    return atoms_list


def train_uma(model, calc, train_atoms, n_epochs, lr, batch_size,
              freeze_backbone, verbose=True) -> float:
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    model.train()

    targets = [float(a.info["REF_energy"]) for a in train_atoms]
    t0 = time.perf_counter()
    n = len(train_atoms)
    for epoch in range(n_epochs):
        order = np.random.permutation(n)
        opt.zero_grad()
        running = 0.0
        for step, idx in enumerate(order):
            atoms = train_atoms[int(idx)]
            e_true = targets[int(idx)]
            e_pred, _ = uma_energy_forces(model, calc, atoms, TASK_NAME)
            loss = (e_pred.squeeze() - e_true) ** 2
            (loss / batch_size).backward()
            running += float(loss)
            if (step + 1) % batch_size == 0 or (step + 1) == n:
                opt.step()
                opt.zero_grad()
        if verbose:
            print(f"    epoch {epoch + 1}/{n_epochs}  mean_loss={running / n:.4f}")
    return time.perf_counter() - t0


def evaluate_split(model, calc, split, verbose=True) -> dict:
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

    dataset = SimplePickleDataset(basedir=DATASET_PATH, label=split, var_config=_VAR_CONFIG)
    model.eval()
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
    p.add_argument("--model-name", default="uma-s-1p2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading OQMD train split …")
    train_atoms = load_atoms("trainset")
    print(f"  train={len(train_atoms)}")

    print(f"Loading UMA ({args.model_name}, {TASK_NAME}) …")
    pu, calc, model = load_trainable_uma(args.model_name, TASK_NAME, args.device)

    print("Fine-tuning …")
    training_sec = train_uma(model, calc, train_atoms, args.epochs, args.lr,
                             args.batch_size, args.freeze_backbone)
    print(f"  Training done in {training_sec:.1f}s — evaluating …")
    test_metrics = evaluate_split(model, calc, "testset")

    result = {args.model_name: {
        "model_name": f"UMA {args.model_name} (FT)",
        "n_epochs": args.epochs,
        "n_train": len(train_atoms),
        "lr": args.lr,
        "freeze_backbone": args.freeze_backbone,
        "training_wall_sec": round(training_sec, 2),
        "testset": test_metrics,
    }}
    out_path = RESULTS_DIR / "uma_finetuned_summary.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"  Test E-MAE = {test_metrics['energy_mae_eV_per_atom']:.4f} eV/atom")


if __name__ == "__main__":
    main()
