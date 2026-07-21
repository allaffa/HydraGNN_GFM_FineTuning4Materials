"""Shared utilities for foundation-model fine-tuning experiments.

Provides:
  - Pickle → ASE Atoms list conversion (with energy/force extraction)
  - Extended XYZ writing (MACE training format)
  - Composition-group helpers (for Wiggle150 relative-energy loss)
  - A simple WallTimer context manager
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Wall-clock timer
# ---------------------------------------------------------------------------

class WallTimer:
    """Context manager that measures elapsed wall-clock time in seconds."""

    def __init__(self):
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Dataset → ASE Atoms
# ---------------------------------------------------------------------------

_DEFAULT_VAR_CONFIG = {
    "type": ["graph"],
    "output_index": [0],
    "output_dim": [1],
    "output_names": ["energy"],
    "graph_feature_names": ["energy"],
    "graph_feature_dims": [1],
    "node_feature_names": ["atomic_number"],
    "node_feature_dims": [1],
    "input_node_features": [0],
    "denormalize_output": False,
}


def _get_gt_value(data) -> float:
    """Extract scalar ground-truth energy from a PyG Data object."""
    y = data.y.detach().cpu()
    return float(y.squeeze()) if y.numel() == 1 else float(y.view(-1)[0])


def pickle_split_to_atoms_list(
    dataset_path: str,
    split: str,
    dataset_type: str,
    max_structures: Optional[int] = None,
    seed: int = 42,
):
    """Convert one split of a pickle dataset to a list of ASE Atoms objects.

    Parameters
    ----------
    dataset_path : str
        Path to the .pickle directory (e.g. ``dataset/md17_mlip.pickle``).
    split : str
        One of ``"trainset"``, ``"valset"``, ``"testset"``.
    dataset_type : str
        ``"mlip"``          — MD17 style: total energy (eV) + forces (eV/Å).
        ``"conformational"`` — Wiggle150 style: relative conformational energy (eV).
        ``"atomization"``   — QM9 style: per-atom atomization energy (eV/atom).
    max_structures : int, optional
        If set, randomly sub-sample the split to at most this many structures
        (deterministic via ``seed``).
    seed : int
        Random seed for sub-sampling.

    Returns
    -------
    list[ase.Atoms]
        ASE Atoms objects with populated ``atoms.info["REF_energy"]`` and
        (for MLIP) ``atoms.arrays["REF_forces"]``.
    """
    import sys
    from pathlib import Path as _Path
    repo_root = _Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "HydraGNN"))

    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
    from utils.uma_calculator import pyg_data_to_ase_atoms

    dataset = SimplePickleDataset(
        basedir=dataset_path, label=split, var_config=_DEFAULT_VAR_CONFIG
    )

    indices = list(range(len(dataset)))
    if max_structures is not None and len(indices) > max_structures:
        rng = random.Random(seed)
        indices = rng.sample(indices, max_structures)
        indices.sort()  # keep reproducible ordering

    atoms_list = []
    for idx in indices:
        data = dataset[idx]
        atoms = pyg_data_to_ase_atoms(data, periodic=False)
        e_raw = _get_gt_value(data)

        if dataset_type == "mlip":
            # Total energy in eV
            atoms.info["REF_energy"] = e_raw
            if hasattr(data, "force") and data.force is not None:
                atoms.arrays["REF_forces"] = data.force.detach().cpu().numpy()

        elif dataset_type == "conformational":
            # Relative conformational energy in eV (already min-shifted per molecule)
            atoms.info["REF_energy"] = e_raw

        elif dataset_type == "atomization":
            # Per-atom atomization energy → total atomization energy
            atoms.info["REF_energy"] = e_raw * len(atoms)

        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

        atoms_list.append(atoms)

    return atoms_list


# ---------------------------------------------------------------------------
# Write extended XYZ (MACE training format)
# ---------------------------------------------------------------------------

def atoms_list_to_xyz(atoms_list: list, output_path: str, has_forces: bool = False):
    """Write an ASE Atoms list to an extended XYZ file for MACE training.

    Reads ``atoms.info["REF_energy"]`` and (optionally) ``atoms.arrays["REF_forces"]``.
    """
    from ase.io import write as ase_write

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # ase_write with format='extxyz' will include all info and arrays fields.
    ase_write(str(path), atoms_list, format="extxyz")
    return str(path)


# ---------------------------------------------------------------------------
# Composition-group helpers (Wiggle150 relative-energy loss)
# ---------------------------------------------------------------------------

def composition_key(atoms) -> tuple:
    """Return a hashable composition fingerprint: sorted tuple of (Z, count)."""
    counts: dict[int, int] = {}
    for z in atoms.get_atomic_numbers():
        counts[int(z)] = counts.get(int(z), 0) + 1
    return tuple(sorted(counts.items()))


def group_by_composition(atoms_list: list) -> dict[tuple, list[int]]:
    """Return {composition_key: [list of indices]} grouping."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, atoms in enumerate(atoms_list):
        groups[composition_key(atoms)].append(i)
    return dict(groups)


# ---------------------------------------------------------------------------
# Timing summary helper
# ---------------------------------------------------------------------------

def timing_dict(
    training_wall_sec: float,
    n_train: int,
    n_epochs: int,
    inference_wall_sec: float,
    n_test: int,
) -> dict:
    """Return a standardised timing sub-dict for benchmark result JSON files."""
    return {
        "training_wall_sec": round(training_wall_sec, 2),
        "training_n_structures": n_train,
        "training_n_epochs": n_epochs,
        "inference_wall_sec": round(inference_wall_sec, 2),
        "inference_structures_per_sec": (
            round(n_test / inference_wall_sec, 2) if inference_wall_sec > 0 else None
        ),
    }
