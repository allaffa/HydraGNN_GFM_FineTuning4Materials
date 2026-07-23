#!/usr/bin/env python3
"""UMA fine-tuning on Wiggle150 (relative conformational energies, energy-only).

Fine-tunes UMA (uma-s-1p2, omol head) on the Wiggle150 train/val splits using a
custom relative-energy loss: within each composition group, predicted energies
are min-shifted and compared to the ground-truth relative energies.  Evaluates
on the test split with the same min-shift convention as the zero-shot benchmark
and saves results to benchmark_results/uma_finetuned_summary.json.

UMA's conservative force head requires ``model.train()`` (create_graph=True) so
the energy graph survives for backprop even though forces are unused here.

Usage
-----
    python examples/wiggle150/run_uma_finetune.py
    python examples/wiggle150/run_uma_finetune.py --epochs 30 --lr 1e-4
    python examples/wiggle150/run_uma_finetune.py --freeze-backbone
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))

from utils.finetune_utils import (
    group_by_composition,
    pickle_split_to_atoms_list,
)
from utils.uma_finetune import load_trainable_uma, uma_energy_forces

KCAL_PER_EV = 23.0609
DATASET_PATH = str(REPO_ROOT / "dataset" / "wiggle150.pickle")
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
TASK_NAME = "omol"


def _composition_key(data) -> tuple:
    z = data.x.detach().cpu().view(-1).long().tolist()
    counts: dict[int, int] = {}
    for zi in z:
        counts[zi] = counts.get(zi, 0) + 1
    return tuple(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Training (relative-energy loss per composition group)
# ---------------------------------------------------------------------------

def train_uma(
    model,
    calc,
    train_atoms,
    n_epochs: int,
    lr: float,
    freeze_backbone: bool,
    verbose: bool = True,
) -> float:
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    model.train()

    groups = group_by_composition(train_atoms)  # {comp_key: [indices]}
    group_keys = list(groups.keys())

    # Pre-extract ground-truth relative energies per structure.
    gt_energy = [float(a.info["REF_energy"]) for a in train_atoms]

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        np.random.shuffle(group_keys)
        running = 0.0
        n_groups = 0
        for key in group_keys:
            idxs = groups[key]
            if len(idxs) < 2:
                continue  # relative loss needs >= 2 conformers

            # UMA's backbone uses gradient checkpointing, so we cannot keep
            # several forward graphs alive at once.  First take a no-grad pass
            # to find the predicted minimum energy (the group reference), then
            # backprop each structure separately against that detached
            # reference and the ground-truth relative energies.
            with torch.no_grad():
                pred_ng = []
                for i in idxs:
                    e_ng, _ = uma_energy_forces(model, calc, train_atoms[i], TASK_NAME)
                    pred_ng.append(float(e_ng.squeeze()))
            min_val = min(pred_ng)
            gt_vals = np.asarray([gt_energy[i] for i in idxs])
            gt_min = float(gt_vals.min())

            opt.zero_grad()
            group_loss = 0.0
            for j, i in enumerate(idxs):
                e_pred, _ = uma_energy_forces(model, calc, train_atoms[i], TASK_NAME)
                pred_rel = e_pred.squeeze() - min_val
                gt_rel = gt_vals[j] - gt_min
                loss = (pred_rel - gt_rel) ** 2 / len(idxs)
                loss.backward()
                group_loss += float(loss)
            opt.step()

            running += group_loss
            n_groups += 1

        if verbose:
            mean_loss = running / max(n_groups, 1)
            print(f"    epoch {epoch + 1}/{n_epochs}  mean_group_loss={mean_loss:.4f}")

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Evaluation (min-shift per composition group, same as zero-shot benchmark)
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

    groups: dict = defaultdict(lambda: {"uma": [], "gt": []})
    _t0 = time.perf_counter()

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = calc

        e_uma = float(atoms.get_potential_energy())
        y = data.y.detach().cpu()
        e_gt = float(y.item() if y.dim() == 0 else y.view(-1)[0])

        comp_key = _composition_key(data)
        groups[comp_key]["uma"].append(e_uma)
        groups[comp_key]["gt"].append(e_gt)

        if verbose and (i + 1) % 50 == 0:
            print(f"    [{split_label}] {i + 1}/{len(dataset)} done …")

    _inference_sec = time.perf_counter() - _t0

    errors = []
    for grp in groups.values():
        uma_arr = np.asarray(grp["uma"])
        gt_arr = np.asarray(grp["gt"])
        uma_rel = uma_arr - uma_arr.min()
        errors.extend((uma_rel - gt_arr).tolist())
    errors = np.asarray(errors)

    return {
        "n_structures": len(errors),
        "n_composition_groups": len(groups),
        "inference_wall_sec": round(_inference_sec, 2),
        "energy_mae_eV":       float(np.abs(errors).mean()),
        "energy_rmse_eV":      float(np.sqrt((errors ** 2).mean())),
        "energy_mae_kcal_mol": float(np.abs(errors).mean()) * KCAL_PER_EV,
        "note": (
            "UMA total energies made relative within each composition group "
            "(min-shifted) to match the Wiggle150 convention."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="uma-s-1p2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--tag", default=None,
                   help="Variant tag; writes uma_finetuned_{tag}_summary.json and suffixes result keys.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading dataset …")
    train_atoms = pickle_split_to_atoms_list(DATASET_PATH, "trainset", "conformational")
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
        freeze_backbone=args.freeze_backbone,
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
    print(f"  Test E-MAE = {test_metrics['energy_mae_eV']:.4f} eV")


if __name__ == "__main__":
    main()
