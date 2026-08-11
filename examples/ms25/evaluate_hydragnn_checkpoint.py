#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import hydragnn
from utils.ensemble_utils import (
    _force_dataset_name_2d,
    _move_batch_to_training_precision,
    get_ensemble,
    load_datasets,
    make_dataloaders,
    setup_distributed_finetuning,
)


def extract_graph_energy(model_output):
    if isinstance(model_output, dict):
        model_output = model_output["graph"]
    if isinstance(model_output, (list, tuple)):
        model_output = model_output[0]
    return model_output.reshape(-1)


def evaluate(system, strategy, repo, batch_size):
    suffix = "mlip" if strategy == "full" else strategy
    log_dir = repo / "examples" / "ms25" / "logs" / f"{system}_{suffix}_seed0"
    config_path = log_dir / f"finetuning_config_{suffix}.json"
    with open(config_path) as config_file:
        config = json.load(config_file)

    training_config = config["NeuralNetwork"]["Training"]
    torch.set_default_dtype(torch.float64 if training_config["precision"] == "fp64" else torch.float32)

    args = SimpleNamespace(
        format="pickle",
        ddstore=False,
        ddstore_width=None,
        shmem=False,
        batch_size=None,
        train_from_scratch=False,
        pretrained_model_ensemble_path=str(repo / "pretrained_model_ensemble"),
        datasetname=f"{system}_mlip_peratom",
        checkpoint_dir=False,
        gfm_2024=False,
    )
    variables = {
        "graph_feature_names": ["energy"],
        "graph_feature_dims": [1],
        "node_feature_names": ["atomic_number", "cartesian_coordinates"],
        "node_feature_dims": [1, 3],
    }

    _, _, testset = load_datasets(args, config, variables)
    _, _, test_loader = make_dataloaders(testset, testset, testset, batch_size=batch_size)
    model = get_ensemble(args, config, test_loader, test_loader, test_loader)

    checkpoint_names = []
    for member in model.module.model_ens:
        member_name = Path(member.module.config["model_type"]).name if hasattr(member.module, "config") else None
        if member_name is None or not (log_dir / member_name / f"{member_name}.pk").exists():
            candidates = sorted(path.name for path in log_dir.iterdir() if path.is_dir() and (path / f"{path.name}.pk").exists())
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one checkpoint member in {log_dir}, found {candidates}")
            member_name = candidates[0]
        hydragnn.utils.model.load_existing_model(member, member_name, path=str(log_dir))
        checkpoint_names.append(member_name)

    model.eval()
    energy_abs_sum = 0.0
    energy_sq_sum = 0.0
    energy_per_atom_abs_sum = 0.0
    force_abs_sum = 0.0
    force_sq_sum = 0.0
    graph_count = 0
    force_component_count = 0
    start = time.time()

    for data in test_loader:
        data = _move_batch_to_training_precision(data, training_config)
        data = _force_dataset_name_2d(data)
        data.pos.requires_grad_(True)

        outputs = model(data)
        member_energies = [extract_graph_energy(output) for output in outputs]
        predicted_energy = torch.stack(member_energies).mean(dim=0)
        predicted_forces = -torch.autograd.grad(
            predicted_energy.sum(), data.pos, create_graph=False
        )[0]

        true_energy = data.energy.reshape(-1).to(predicted_energy.dtype)
        energy_error = predicted_energy - true_energy
        natoms = (data.ptr[1:] - data.ptr[:-1]).to(predicted_energy.dtype)
        force_error = predicted_forces - data.forces.to(predicted_forces.dtype)

        energy_abs_sum += energy_error.abs().sum().item()
        energy_sq_sum += energy_error.square().sum().item()
        energy_per_atom_abs_sum += (energy_error.abs() / natoms).sum().item()
        force_abs_sum += force_error.abs().sum().item()
        force_sq_sum += force_error.square().sum().item()
        graph_count += true_energy.numel()
        force_component_count += force_error.numel()

    result = {
        "test_mae_eV": energy_abs_sum / graph_count,
        "test_rmse_eV": (energy_sq_sum / graph_count) ** 0.5,
        "test_mae_eV_per_atom": energy_per_atom_abs_sum / graph_count,
        "test_force_mae_eV_A": force_abs_sum / force_component_count,
        "test_force_rmse_eV_A": (force_sq_sum / force_component_count) ** 0.5,
        "test_structures": graph_count,
        "inference_wall_sec": time.time() - start,
        "checkpoint_members": checkpoint_names,
        "training_force_weight": config["NeuralNetwork"]["Architecture"].get("force_weight", 0.0),
    }
    output_path = log_dir / "benchmark_results" / "checkpoint_benchmark_summary.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(result, output_file, indent=2)
    print(f"{system}: E-MAE={result['test_mae_eV']:.6f} eV, "
          f"E-MAE/atom={result['test_mae_eV_per_atom'] * 1000:.3f} meV/atom, "
          f"F-MAE={result['test_force_mae_eV_A']:.6f} eV/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--strategy", default="full")
    parser.add_argument("--batch-size", type=int, default=2)
    parsed = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    setup_distributed_finetuning()
    evaluate(parsed.system, parsed.strategy, repository, parsed.batch_size)