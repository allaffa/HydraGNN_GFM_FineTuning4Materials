"""MACE foundation model calculator utilities.

Wraps the mace-torch ASE calculators (``mace_off``, ``mace_polar``, ``mace_mp``)
as lightweight helpers used by the MACE benchmark scripts in this project.

Prerequisites
-------------
    pip install mace-torch>=0.3.16

Available model IDs  (``MACE_MODELS`` keys)
--------------------------------------------
    mace_off_medium   — MACE-OFF23 medium, SPICE v1, ωB97M+D3, 10 organic elements
    mace_polar_m      — MACE-POLAR-1 polar-1-m, OMol25, ωB97M-V, 83 elements
    mace_mh1_omol     — MACE-MH-1 omol head, ωB97M-VV10, 89 elements

The ``pyg_data_to_ase_atoms`` helper is re-exported from
:mod:`utils.uma_calculator` so callers only need one import.

References
----------
Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks
  for Fast and Accurate Force Fields", NeurIPS 2022.
Kovacs et al., "MACE-OFF23: Transferable Machine Learning Force Fields for
  Organic Molecules", arXiv:2312.15211.
Batatia et al., "Cross Learning between Electronic Structure Theories …
  (MACE-MH-1)", arXiv:2510.25380.
Batatia et al., "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
  for Molecular Chemistry", arXiv:2602.19411.
"""

from __future__ import annotations

import torch

# Re-export pyg_data_to_ase_atoms so benchmark scripts only need one import.
from utils.uma_calculator import pyg_data_to_ase_atoms  # noqa: F401

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

#: Catalogue of supported MACE foundation models.
#:
#: Keys are short model IDs used as ``--models`` arguments in the benchmark
#: scripts.  Each entry describes the MACE family, model variant, optional
#: head, and whether ``atoms.info`` charge/spin must be set before inference.
MACE_MODELS: dict[str, dict] = {
    "mace_off_medium": {
        "label": "MACE-OFF23 (medium)",
        "family": "mace_off",
        "model": "medium",
        "head": None,
        # MACE-OFF23 is a neutral organic force field; charge/spin not required.
        "needs_charge_spin": False,
        "training_data": "SPICE v1",
        "level_of_theory": "ωB97M+D3",
        "n_elements": 10,
    },
    "mace_polar_m": {
        "label": "MACE-POLAR-1 (polar-1-m)",
        "family": "mace_polar",
        "model": "polar-1-m",
        "head": None,
        # MACE-POLAR-1 learns charge/spin densities — must set atoms.info.
        "needs_charge_spin": True,
        "training_data": "OMol25",
        "level_of_theory": "ωB97M-V",
        "n_elements": 83,
    },
    "mace_mh1_omol": {
        "label": "MACE-MH-1 (omol head)",
        "family": "mace_mp",
        "model": "mh-1",
        "head": "omol",
        # The omol head was trained on OMOL25 (neutral subset); set charge/spin
        # to avoid potential warnings or default fallbacks inside mace_mp.
        "needs_charge_spin": True,
        "training_data": "OMAT+OMOL+SPICE+MPTraj+OC20+RGD1+MATPES",
        "level_of_theory": "ωB97M-VV10 (omol head)",
        "n_elements": 89,
    },
}


# ---------------------------------------------------------------------------
# Calculator factory
# ---------------------------------------------------------------------------

def build_mace_calculator(
    model_id: str,
    device: str | None = None,
    default_dtype: str = "float64",
):
    """Build and return a MACE ASE calculator for the requested model.

    Parameters
    ----------
    model_id:
        One of the keys in :data:`MACE_MODELS` (e.g. ``"mace_off_medium"``).
    device:
        Target device string (``"cuda"``, ``"cpu"``, …).  If *None*, uses CUDA
        when available, otherwise CPU.
    default_dtype:
        Floating-point precision (``"float64"`` or ``"float32"``).  Use
        ``"float64"`` for benchmark accuracy.

    Returns
    -------
    ASE Calculator
        An ASE-compatible calculator with ``get_potential_energy()`` and
        ``get_forces()`` methods.

    Raises
    ------
    ValueError
        If *model_id* is not in :data:`MACE_MODELS`.
    ImportError
        If ``mace-torch`` is not installed.
    """
    if model_id not in MACE_MODELS:
        raise ValueError(
            f"Unknown model_id {model_id!r}.  Choose from: "
            + ", ".join(MACE_MODELS)
        )

    try:
        from mace.calculators import mace_mp, mace_off, mace_polar
    except ImportError as exc:
        raise ImportError(
            "mace-torch is required for MACE benchmarks.  Install it with:\n"
            "    pip install mace-torch>=0.3.16"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MACE_MODELS[model_id]

    if cfg["family"] == "mace_off":
        calc = mace_off(
            model=cfg["model"],
            device=device,
            default_dtype=default_dtype,
        )
    elif cfg["family"] == "mace_polar":
        calc = mace_polar(
            model=cfg["model"],
            device=device,
            default_dtype=default_dtype,
        )
    elif cfg["family"] == "mace_mp":
        calc = mace_mp(
            model=cfg["model"],
            device=device,
            default_dtype=default_dtype,
            head=cfg["head"],
        )
    else:
        raise ValueError(f"Unknown MACE family {cfg['family']!r}")

    return calc
