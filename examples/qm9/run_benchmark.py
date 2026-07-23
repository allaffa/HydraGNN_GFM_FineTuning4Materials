#!/usr/bin/env python3
"""QM9 energy-only benchmark: fine-tuning strategies vs scratch training.

Predicts per-atom atomization energy (eV/atom), mean-shifted.
No force computation — uses standard graph-level MAE loss.
"""

import sys, os, json, copy, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import mpi4py
mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

import hydragnn
from hydragnn.utils.distributed import setup_ddp, get_device
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from hydragnn.utils.input_config_parsing.config_utils import (
    update_config_edge_dim,
    update_config_equivariance,
)
from hydragnn.train.train_validate_test import (
    resolve_precision,
    move_batch_to_device,
    get_autocast_and_scaler,
)

from utils.update_model import update_model as _update_model
from utils.ensemble_utils import (
    update_GFM_2024_checkpoint,
    get_distributed_model_find_unused,
    _force_dataset_name_2d,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRETRAINED_DIR = str(
    REPO_ROOT / "pretrained_model_ensemble"
    / "multidataset_hpo-BEST6-fp64"
)
FT_CONFIG_PATH = str(REPO_ROOT / "examples" / "qm9" / "finetuning_config_energy.json")
FT_CONFIG_ANI1X_PATH = str(REPO_ROOT / "examples" / "qm9" / "finetuning_config_energy_ani1x.json")
DATASET_DIR = str(REPO_ROOT / "dataset" / "qm9_energy.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "qm9" / "benchmark_results")

NUM_EPOCHS = 10
WARMUP_EPOCHS = 0
BACKBONE_LR_MULT = 0.1
BATCH_SIZE = 32

# Per-strategy learning rates
STRATEGY_LR = {
    "scratch": 1e-3,
    "unfrozen": 1e-3,
    "frozen": 1e-3,
    "ani1x_recycled": 1e-3,
    "qm7x_recycled": 1e-3,
}

_DEFAULT_GRAPH_ATTR = torch.tensor([0.0, 1.0])

ARCH_DEFAULTS = {
    "global_attn_engine": None, "global_attn_type": None, "global_attn_heads": 0,
    "pe_dim": 0, "pna_deg": None, "freeze_conv_layers": False,
    "initial_bias": None, "activation_function": "relu", "SyncBatchNorm": False,
    "radius": None, "radial_type": None, "distance_transform": None,
    "num_gaussians": None, "num_filters": None, "envelope_exponent": None,
    "num_after_skip": None, "num_before_skip": None, "basis_emb_size": None,
    "int_emb_size": None, "out_emb_size": None, "num_radial": None,
    "num_spherical": None, "correlation": None, "max_ell": None,
    "node_max_ell": None, "avg_num_neighbors": None,
}

KCAL_PER_EV = 23.0609


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_graph_attr(batch):
    if not hasattr(batch, "graph_attr") or batch.graph_attr is None:
        ng = int(batch.batch.max().item() + 1) if hasattr(batch, "batch") else 1
        batch.graph_attr = _DEFAULT_GRAPH_ATTR.unsqueeze(0).expand(ng, -1).clone()
    return batch


def load_ft_config():
    with open(FT_CONFIG_PATH) as f:
        return json.load(f)


def load_ft_config_ani1x():
    with open(FT_CONFIG_ANI1X_PATH) as f:
        return json.load(f)


def load_pretrained_config():
    with open(os.path.join(PRETRAINED_DIR, "config.json")) as f:
        cfg = json.load(f)
    arch = cfg["NeuralNetwork"]["Architecture"]
    for k, v in ARCH_DEFAULTS.items():
        arch.setdefault(k, v)
    arch.update(update_config_edge_dim(arch))
    arch.update(update_config_equivariance(arch))
    training = cfg["NeuralNetwork"]["Training"]
    training.setdefault("compute_grad_energy", False)
    training.setdefault("conv_checkpointing", False)
    return cfg


def make_dataloaders(ft_config):
    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = ["energy"]
    var_config["graph_feature_dims"] = [1]
    var_config["node_feature_names"] = ["atomic_number"]
    var_config["node_feature_dims"] = [1]
    var_config["input_node_features"] = [0]

    trainset = SimplePickleDataset(basedir=DATASET_DIR, label="trainset", var_config=var_config)
    valset = SimplePickleDataset(basedir=DATASET_DIR, label="valset", var_config=var_config)
    testset = SimplePickleDataset(basedir=DATASET_DIR, label="testset", var_config=var_config)
    return hydragnn.preprocess.create_dataloaders(trainset, valset, testset, BATCH_SIZE)


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------
def build_model_from_pretrained(pretrained_config, ft_config, freeze=False):
    """Build model from pretrained backbone, swap to single energy head."""
    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"], verbosity=0,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity=0)

    update_GFM_2024_checkpoint(
        model,
        os.path.basename(PRETRAINED_DIR),
        path=os.path.dirname(PRETRAINED_DIR),
    )

    model = model.module
    model = _update_model(model, ft_config)
    if freeze:
        model._freeze_conv()

    return model


def build_scratch_model(pretrained_config, ft_config):
    """Build model with random weights using the same architecture."""
    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"], verbosity=0,
    )
    model = _update_model(model, ft_config)
    return model


ANI1X_BRANCH = 1   # branch-1  = ANI1x in the pretrained 16-head model
QM7X_BRANCH  = 14  # branch-14 = QM7X  in the pretrained 16-head model

BRANCH_LABEL = {ANI1X_BRANCH: "ANI1x", QM7X_BRANCH: "QM7X"}


def build_model_with_recycled_head(pretrained_config, ft_config,
                                   source_branch=ANI1X_BRANCH, freeze=False):
    """Build model from pretrained backbone, recycling a specific head."""
    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"], verbosity=0,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity=0)

    update_GFM_2024_checkpoint(
        model,
        os.path.basename(PRETRAINED_DIR),
        path=os.path.dirname(PRETRAINED_DIR),
    )

    model = model.module

    src_tag = f"branch-{source_branch}"
    saved_shared_state = model.graph_shared[src_tag].state_dict()

    saved_head_state = None
    for head_dict in model.heads_NN:
        if src_tag in head_dict:
            saved_head_state = head_dict[src_tag].state_dict()
            break

    n_shared = sum(1 for _ in saved_shared_state)
    n_head = sum(1 for _ in saved_head_state) if saved_head_state else 0
    label = BRANCH_LABEL.get(source_branch, f"branch-{source_branch}")
    print(f"  Recycling pretrained {src_tag} ({label}) → branch-0")
    print(f"    Saved {n_shared} shared params, {n_head} head params")

    model = _update_model(model, ft_config)

    with torch.no_grad():
        model.graph_shared["branch-0"].load_state_dict(saved_shared_state)
        if saved_head_state is not None:
            model.heads_NN[0]["branch-0"].load_state_dict(saved_head_state)

    if freeze:
        model._freeze_conv()

    return model


# ---------------------------------------------------------------------------
# Optimizer / Scheduler helpers
# ---------------------------------------------------------------------------
def make_param_groups(model, base_lr, strategy):
    use_differential = False

    if strategy == "frozen":
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": base_lr}]

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = name.replace("module.", "", 1) if name.startswith("module.") else name
        if "heads_NN" in clean or "graph_shared" in clean:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if use_differential:
        bb_lr = base_lr * BACKBONE_LR_MULT
        print(f"    Differential lr: backbone={bb_lr:.2e}, heads={base_lr:.2e}")
        return [
            {"params": backbone_params, "lr": bb_lr},
            {"params": head_params, "lr": base_lr},
        ]
    else:
        return [{"params": backbone_params + head_params, "lr": base_lr}]


def make_scheduler(optimizer, num_epochs, warmup_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
    )
    constant = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=num_epochs - warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, constant], milestones=[warmup_epochs],
    )


# ---------------------------------------------------------------------------
# Training loop — energy only (MAE loss)
# ---------------------------------------------------------------------------
def train_loop(model, train_loader, val_loader, num_epochs, optimizer, precision,
               scheduler=None):
    prec, param_dtype, _ = resolve_precision(precision)
    autocast_ctx, scaler = get_autocast_and_scaler(prec)
    device = get_device()

    energy_mae_hist = []

    for epoch in range(num_epochs):
        os.environ["HYDRAGNN_EPOCH"] = str(epoch)

        # ---------- Train ----------
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = _force_dataset_name_2d(batch)
            batch = _ensure_graph_attr(batch)
            batch = move_batch_to_device(batch, param_dtype)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                pred = model(batch)
                # pred is a list; pred[0] = [B, 1] for graph-level head
                e_pred = pred[0].view(-1)
                e_true = batch.y.view(-1).to(e_pred.dtype)
                loss = F.l1_loss(e_pred, e_true)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.detach().cpu())

        mean_train_loss = epoch_loss / len(train_loader)

        # ---------- Validate ----------
        model.eval()
        all_pred = []
        all_true = []
        all_natoms = []

        with torch.no_grad():
            for batch in val_loader:
                batch = _force_dataset_name_2d(batch)
                batch = _ensure_graph_attr(batch)
                batch = move_batch_to_device(batch, param_dtype)

                pred = model(batch)
                e_pred = pred[0].view(-1).float()
                e_true = batch.y.view(-1).float()

                # Number of atoms per graph in batch
                natoms = torch.bincount(batch.batch).float()

                all_pred.append(e_pred.cpu())
                all_true.append(e_true.cpu())
                all_natoms.append(natoms.cpu())

        all_pred = torch.cat(all_pred)
        all_true = torch.cat(all_true)
        all_natoms = torch.cat(all_natoms)

        # Per-atom MAE: |E_pred - E_true| / N_atoms, averaged over samples
        peratom_ae = (all_pred - all_true).abs() / all_natoms
        energy_mae = float(peratom_ae.mean())
        energy_mae_hist.append(energy_mae)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_parts = " ".join(
                f"lr{i}={pg['lr']:.2e}" for i, pg in enumerate(optimizer.param_groups)
            )
            print(
                f"    Epoch {epoch+1:4d}/{num_epochs}"
                f"  Loss: {mean_train_loss:.6f}"
                f"  Val E-MAE: {energy_mae:.6f} eV/atom"
                f"  ({energy_mae * KCAL_PER_EV:.4f} kcal/(mol·atom))"
                f"  [{lr_parts}]"
            )

    return {"energy_mae": energy_mae_hist}


def evaluate_energy(model, loader, param_dtype):
    """Run energy inference over a loader; return per-atom energy MAE (eV/atom)."""
    model.eval()
    all_pred = []
    all_true = []
    all_natoms = []

    with torch.no_grad():
        for batch in loader:
            batch = _force_dataset_name_2d(batch)
            batch = _ensure_graph_attr(batch)
            batch = move_batch_to_device(batch, param_dtype)

            pred = model(batch)
            e_pred = pred[0].view(-1).float()
            e_true = batch.y.view(-1).float()
            natoms = torch.bincount(batch.batch).float()

            all_pred.append(e_pred.cpu())
            all_true.append(e_true.cpu())
            all_natoms.append(natoms.cpu())

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    all_natoms = torch.cat(all_natoms)
    peratom_ae = (all_pred - all_true).abs() / all_natoms
    return float(peratom_ae.mean())


# ---------------------------------------------------------------------------
# Run single experiment
# ---------------------------------------------------------------------------
def run_experiment(strategy, ft_config, pretrained_config, train_loader, val_loader,
                   test_loader=None):
    precision = "fp64"
    prec, param_dtype, _ = resolve_precision(precision)

    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy}  |  Energy only (total, eV; reported per-atom)")
    print(f"{'='*60}")

    if strategy in ("frozen", "unfrozen"):
        freeze = (strategy == "frozen")
        model = build_model_from_pretrained(pretrained_config, ft_config, freeze=freeze)
    elif strategy == "ani1x_recycled":
        model = build_model_with_recycled_head(pretrained_config, ft_config,
                                               source_branch=ANI1X_BRANCH, freeze=False)
    elif strategy == "qm7x_recycled":
        model = build_model_with_recycled_head(pretrained_config, ft_config,
                                               source_branch=QM7X_BRANCH, freeze=False)
    else:
        model = build_scratch_model(pretrained_config, ft_config)

    model = model.to(dtype=param_dtype)
    model = get_distributed_model_find_unused(model, verbosity=0)

    lr = STRATEGY_LR.get(strategy, ft_config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    wd = ft_config["NeuralNetwork"]["Training"]["Optimizer"].get("weight_decay", 0.0)

    param_groups = make_param_groups(model, lr, strategy)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    scheduler = None
    print(f"    Scheduler: constant lr={lr:.1e} for {NUM_EPOCHS} epochs")

    _train_t0 = time.perf_counter()
    history = train_loop(model, train_loader, val_loader, NUM_EPOCHS, optimizer,
                         precision, scheduler=scheduler)
    training_wall_sec = time.perf_counter() - _train_t0

    best_e = min(history["energy_mae"])
    best_e_ep = history["energy_mae"].index(best_e) + 1
    print(f"  Best Val Energy MAE: {best_e:.6f} eV/atom"
          f"  ({best_e * KCAL_PER_EV:.4f} kcal/(mol·atom)) at epoch {best_e_ep}")
    print(f"  Training wall-clock : {training_wall_sec:.1f} s")

    history["training_wall_sec"] = training_wall_sec

    # ---------- Timed test-set inference ----------
    if test_loader is not None:
        _infer_t0 = time.perf_counter()
        test_e_mae = evaluate_energy(model, test_loader, param_dtype)
        inference_wall_sec = time.perf_counter() - _infer_t0
        print(f"  Test  Energy MAE    : {test_e_mae:.6f} eV/atom"
              f"  ({test_e_mae * KCAL_PER_EV:.4f} kcal/(mol·atom))")
        print(f"  Inference wall-clock: {inference_wall_sec:.2f} s")
        history["test_energy_mae_eV_atom"] = test_e_mae
        history["inference_wall_sec"] = inference_wall_sec

    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(results, output_dir):
    labels = {
        "frozen": "Fine-tuning (frozen)",
        "unfrozen": "Fine-tuning (unfrozen)",
        "scratch": "From scratch",
        "ani1x_recycled": "ANI1x head recycled",
        "qm7x_recycled": "QM7X head recycled",
    }
    colors = {
        "frozen": "#1f77b4", "unfrozen": "#ff7f0e",
        "scratch": "#2ca02c", "ani1x_recycled": "#d62728",
        "qm7x_recycled": "#9467bd",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for strategy in ("frozen", "unfrozen", "scratch", "ani1x_recycled", "qm7x_recycled"):
        if strategy not in results:
            continue
        hist = results[strategy]["energy_mae"]
        epochs = list(range(1, len(hist) + 1))
        ax.plot(epochs, hist, label=labels[strategy], color=colors[strategy], lw=1.2)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Energy MAE (eV/atom)", fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("QM9 — Per-atom Atomization Energy", fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "qm9_energy_validation.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    world_size, world_rank = setup_ddp()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ft_config_template = load_ft_config_ani1x()
    pretrained_config = load_pretrained_config()

    train_loader, val_loader, test_loader = make_dataloaders(
        copy.deepcopy(ft_config_template)
    )

    strategies = ["scratch", "unfrozen", "ani1x_recycled", "qm7x_recycled", "frozen"]
    results = {}

    for strategy in strategies:
        ft_config = copy.deepcopy(ft_config_template)
        results[strategy] = run_experiment(
            strategy, ft_config, pretrained_config, train_loader, val_loader,
            test_loader=test_loader,
        )

    # Save summary
    summary = {}
    for strategy, hist in results.items():
        best_e = min(hist["energy_mae"])
        summary[strategy] = {
            "best_energy_mae_eV_atom": best_e,
            "best_energy_mae_kcal_mol_atom": best_e * KCAL_PER_EV,
            "best_energy_epoch": hist["energy_mae"].index(best_e) + 1,
            "training_wall_sec": round(hist.get("training_wall_sec", 0.0), 2),
            "inference_wall_sec": (
                round(hist["inference_wall_sec"], 2)
                if "inference_wall_sec" in hist else None
            ),
            "test_energy_mae_eV_atom": hist.get("test_energy_mae_eV_atom"),
        }

    summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    history_path = os.path.join(OUTPUT_DIR, "val_histories.json")
    with open(history_path, "w") as f:
        json.dump(results, f)
    print(f"Histories saved to {history_path}")

    plot_curves(results, OUTPUT_DIR)

    # Print table
    print(f"\n{'Strategy':<18s}  {'E-MAE (eV/atom)':>15s}  {'E-MAE (kcal/(mol·atom))':>24s}  {'Ep':>4s}")
    print("-" * 68)
    for strategy in strategies:
        s = summary[strategy]
        print(
            f"{strategy:<18s}"
            f"  {s['best_energy_mae_eV_atom']:15.6f}"
            f"  {s['best_energy_mae_kcal_mol_atom']:24.4f}"
            f"  {s['best_energy_epoch']:4d}"
        )


if __name__ == "__main__":
    main()
