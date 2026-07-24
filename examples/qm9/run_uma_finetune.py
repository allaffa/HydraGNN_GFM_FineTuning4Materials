#!/usr/bin/env python3
"""UMA fine-tuning on QM9 (per-atom atomization energy, energy-only).

Fine-tunes UMA (uma-s-1p2, omol head) on a random subset of the QM9 train
split (energy-only; QM9 has no forces).  The training target is the total
atomization energy (per-atom U0 × n_atoms).  Evaluation uses the per-element
linear reference correction on the test split, matching the zero-shot
benchmark, and saves results to benchmark_results/uma_finetuned_summary.json.

The full QM9 train split (~91k structures) is too slow to fine-tune on CPU, so
a deterministic random subset (default 5000, seed 42) is used.

UMA's conservative force head requires ``model.train()`` (create_graph=True) so
the energy graph survives for backprop even though forces are unused here.

Usage
-----
    python examples/qm9/run_uma_finetune.py
    python examples/qm9/run_uma_finetune.py --n-train 5000 --epochs 10
    python examples/qm9/run_uma_finetune.py --freeze-backbone
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
DATASET_PATH = str(REPO_ROOT / "dataset" / "qm9_energy.pickle")
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
TASK_NAME = "omol"


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
# Training (energy-only, total atomization energy target)
# ---------------------------------------------------------------------------

def train_uma(
    model,
    calc,
    train_atoms,
    n_epochs: int,
    lr: float,
    batch_size: int,
    freeze_backbone: bool,
    lora: bool = False,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    verbose: bool = True,
) -> float:
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

    # Centre targets to stabilise energy-only training; the constant offset is
    # irrelevant because evaluation refits a per-element linear reference.
    targets = np.asarray([float(a.info["REF_energy"]) for a in train_atoms])
    target_mean = float(targets.mean())

    t0 = time.perf_counter()
    n = len(train_atoms)
    for epoch in range(n_epochs):
        order = np.random.permutation(n)
        opt.zero_grad()
        running = 0.0
        for step, idx in enumerate(order):
            atoms = train_atoms[int(idx)]
            e_true = float(targets[int(idx)]) - target_mean

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


# ---------------------------------------------------------------------------
# Evaluation (per-element linear reference correction, same as zero-shot)
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

    e_preds_total = []
    compositions = []
    n_atoms_list = []
    gt_per_atom = []
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
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
            "UMA total energy corrected by per-element linear reference "
            "energies (least-squares fit on evaluation split), divided by "
            "n_atoms, compared to QM9 per-atom atomization energy."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="uma-s-1p2")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-train", type=int, default=5000,
                   help="Random training subset size (default: 5000, seed 42).")
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading dataset (subsampled) …")
    train_atoms = pickle_split_to_atoms_list(
        DATASET_PATH, "trainset", "atomization",
        max_structures=args.n_train, seed=args.seed,
    )
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
        freeze_backbone=args.freeze_backbone,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    print(f"  Training done in {training_sec:.1f}s")

    print("Evaluating on test set …")
    test_metrics = evaluate_split(model, calc, "testset")

    result = {
        (f"{args.model_name} [{args.tag}]" if args.tag else args.model_name): {
            "model_name":        f"UMA {args.model_name} (fine-tuned)",
            "n_epochs":          args.epochs,
            "n_train":           len(train_atoms),
            "lr":                args.lr,
            "freeze_backbone":   args.freeze_backbone,
            "lora":              args.lora,
            "training_wall_sec": round(training_sec, 2),
            "testset":           test_metrics,
        }
    }

    _summary_name = (
        f"uma_finetuned_{args.tag}_summary.json" if args.tag
        else "uma_finetuned_summary.json"
    )
    out_path = RESULTS_DIR / _summary_name
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {out_path}")
    print(
        f"  Test E/atom-MAE = {test_metrics['energy_per_atom_mae_eV']:.4f} eV/atom"
    )


if __name__ == "__main__":
    main()
