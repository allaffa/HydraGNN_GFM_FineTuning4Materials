"""UMA (Universal Model for Atoms) calculator utilities.

Wraps Meta's fairchem-core ``FAIRChemCalculator`` as a lightweight helper used
by the UMA benchmark scripts in this project.  The integration pattern follows
the matsim-agents project (https://github.com/ORNL/matsim-agents).

Prerequisites
-------------
    pip install fairchem-core>=2.20

Authentication
--------------
UMA weights require a free Hugging Face account with accepted license terms:
  1. Visit https://huggingface.co/facebook/UMA and accept the FAIR Chemistry License.
  2. Run: huggingface-cli login

Available model names
---------------------
    uma-s-1p2   — small, fastest, ~6.6 M active params  (recommended)
    uma-s-1p1   — earlier small model
    uma-m-1p1   — medium, most accurate, ~50 M active params

Task names (task_name argument)
--------------------------------
    omat        — inorganic bulk materials  (periodic, DFT-PBE)
    omol        — molecules & MOFs          (non-periodic, DFT-ωB97X-D)
    oc20        — surface catalysis with adsorbates
    odac        — metal-organic frameworks
    omc         — molecular crystals

Reference
---------
Wood et al., "UMA: A Family of Universal Models for Atoms", arXiv:2506.23971
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Calculator factory
# ---------------------------------------------------------------------------

def build_uma_calculator(
    model_name: str = "uma-s-1p2",
    task_name: Literal["omat", "omol", "oc20", "odac", "omc"] = "omol",
    device: str | None = None,
    local_cache: str | None = None,
):
    """Build and return a ``FAIRChemCalculator`` wrapping a UMA checkpoint.

    Parameters
    ----------
    model_name:
        Pretrained model identifier (e.g. ``"uma-s-1p2"``).  Can also be an
        absolute path to a local ``.pt`` checkpoint.
    task_name:
        UMA task head.  Use ``"omol"`` for molecules/non-periodic systems and
        ``"omat"`` for inorganic periodic materials.
    device:
        Target device string (``"cuda"``, ``"cpu"``, …).  If *None*, uses CUDA
        when available, otherwise CPU.
    local_cache:
        Directory used by ``model_name_to_local_file`` to cache downloaded
        checkpoints.  Defaults to ``~/.cache/fairchem``.

    Returns
    -------
    FAIRChemCalculator
        ASE-compatible calculator with ``energy`` and ``forces`` properties.
    """
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.models.model_registry import model_name_to_local_file
    except ImportError as exc:
        raise ImportError(
            "fairchem-core is required for UMA benchmarks.  Install it with:\n"
            "    pip install fairchem-core>=2.20\n"
            "Then accept the UMA license at https://huggingface.co/facebook/UMA\n"
            "and authenticate: huggingface-cli login"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if local_cache is None:
        local_cache = os.path.expanduser("~/.cache/fairchem")

    # If the caller passed a file path, use it directly; otherwise resolve name.
    if os.path.isfile(model_name):
        checkpoint_path = model_name
    else:
        checkpoint_path = model_name_to_local_file(
            model_name, local_cache=local_cache
        )

    calc = FAIRChemCalculator(
        checkpoint_path=checkpoint_path,
        task_name=task_name,
        seed=0,
        device=device,
    )
    return calc


# ---------------------------------------------------------------------------
# Data-conversion helpers
# ---------------------------------------------------------------------------

def pyg_data_to_ase_atoms(data, periodic: bool = False):
    """Convert a HydraGNN/PyG ``Data`` object to an ASE ``Atoms`` instance.

    Parameters
    ----------
    data:
        A ``torch_geometric.data.Data`` object produced by the HydraGNN
        pre-processing scripts in this project.  Must have:

        * ``data.pos``  — atomic positions in Å, shape ``[N, 3]``
        * ``data.x``    — node features, first column is atomic number
                          (float), shape ``[N, F]``

        Optional (for periodic systems):

        * ``data.cell`` — unit-cell tensor, shape ``[3, 3]`` or ``[1, 9]``
        * ``data.pbc``  — periodic boundary flags, shape ``[3]``

    periodic:
        Force periodic boundary conditions even if ``data.pbc`` is absent.
        Set to ``True`` for bulk-material datasets.

    Returns
    -------
    ase.Atoms
    """
    from ase import Atoms

    pos = data.pos.detach().cpu().numpy().astype(np.float64)
    z = data.x[:, 0].detach().cpu().long().numpy()

    cell = None
    pbc = False

    if periodic or (hasattr(data, "pbc") and data.pbc is not None and data.pbc.any()):
        pbc = True
        if hasattr(data, "cell") and data.cell is not None:
            c = data.cell.detach().cpu().numpy()
            cell = c.reshape(3, 3)

    atoms = Atoms(numbers=z, positions=pos, cell=cell, pbc=pbc)
    return atoms


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_uma_on_dataset(
    dataset,
    model_name: str = "uma-s-1p2",
    task_name: Literal["omat", "omol", "oc20", "odac", "omc"] = "omol",
    device: str | None = None,
    periodic: bool = False,
    compute_forces: bool = False,
    energy_attr: str = "energy",
    forces_attr: str = "forces",
    verbose: bool = True,
):
    """Run UMA inference over a HydraGNN-formatted dataset and return metrics.

    Parameters
    ----------
    dataset:
        Any iterable of ``torch_geometric.data.Data`` objects.
    model_name, task_name, device:
        Forwarded to :func:`build_uma_calculator`.
    periodic:
        Passed to :func:`pyg_data_to_ase_atoms`.
    compute_forces:
        If ``True``, also collect force MAE.
    energy_attr:
        Attribute name on each ``Data`` object holding the ground-truth energy
        scalar (default ``"energy"``).
    forces_attr:
        Attribute name on each ``Data`` object holding the ground-truth forces
        tensor of shape ``[N, 3]`` (default ``"forces"``).
    verbose:
        Print progress every 100 structures.

    Returns
    -------
    dict with keys:
        ``energy_mae`` (float, eV),
        ``energy_rmse`` (float, eV),
        ``force_mae`` (float, eV/Å) — only when ``compute_forces=True``,
        ``force_rmse`` (float, eV/Å) — only when ``compute_forces=True``,
        ``n_structures`` (int),
    """
    calc = build_uma_calculator(
        model_name=model_name, task_name=task_name, device=device
    )

    energy_errors = []
    force_errors_flat = []

    for i, data in enumerate(dataset):
        atoms = pyg_data_to_ase_atoms(data, periodic=periodic)
        atoms.calc = calc

        # UMA inference
        e_pred = atoms.get_potential_energy()  # eV

        # Ground-truth energy
        e_true_tensor = getattr(data, energy_attr, None)
        if e_true_tensor is None:
            raise AttributeError(
                f"Data object has no attribute '{energy_attr}'. "
                "Check the energy_attr argument."
            )
        e_true = float(e_true_tensor.detach().cpu().squeeze())

        energy_errors.append(e_pred - e_true)

        if compute_forces:
            f_pred = atoms.get_forces()  # shape [N, 3], eV/Å
            f_true_tensor = getattr(data, forces_attr, None)
            if f_true_tensor is None:
                raise AttributeError(
                    f"Data object has no attribute '{forces_attr}'. "
                    "Check the forces_attr argument."
                )
            f_true = f_true_tensor.detach().cpu().numpy()  # [N, 3]
            force_errors_flat.append((f_pred - f_true).ravel())

        if verbose and (i + 1) % 100 == 0:
            print(f"  Evaluated {i + 1} structures …")

    energy_errors = np.asarray(energy_errors)
    results = {
        "energy_mae": float(np.abs(energy_errors).mean()),
        "energy_rmse": float(np.sqrt((energy_errors ** 2).mean())),
        "n_structures": len(energy_errors),
    }

    if compute_forces and force_errors_flat:
        flat = np.concatenate(force_errors_flat)
        results["force_mae"] = float(np.abs(flat).mean())
        results["force_rmse"] = float(np.sqrt((flat ** 2).mean()))

    return results
