"""Shared helpers for UMA (fairchem) fine-tuning experiments.

UMA uses a conservative energy-and-force head that computes forces via
autograd during the forward pass.  Training therefore requires
``model.train()`` so the head sets ``create_graph=True`` (double backprop);
otherwise the energy graph is freed by the internal force autograd and the
outer ``loss.backward()`` fails.

Typical usage
-------------
    pu, calc, model = load_trainable_uma("uma-s-1p2", "omol", "cpu")
    model.train()
    e, f = uma_energy_forces(model, calc, atoms)   # differentiable
    loss = ...
    loss.backward(); opt.step(); opt.zero_grad()

    # Evaluation reuses the same in-memory calculator:
    model.eval()
    atoms.calc = calc
    e_pred = atoms.get_potential_energy()
"""

from __future__ import annotations

import torch


def load_trainable_uma(
    model_name: str = "uma-s-1p2",
    task_name: str = "omol",
    device: str = "cpu",
):
    """Load a UMA checkpoint exposing a trainable ``HydraModel``.

    Returns
    -------
    (predict_unit, calculator, model) tuple where
        * ``predict_unit`` is the ``MLIPPredictUnit`` (used for evaluation),
        * ``calculator`` is a ``FAIRChemCalculator`` wrapping it (a2g + eval),
        * ``model`` is the trainable ``HydraModel`` (``pu.model.module``).
    """
    try:
        from fairchem.core import pretrained_mlip
        from fairchem.core.calculate import FAIRChemCalculator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fairchem-core is required for UMA fine-tuning.  Install it with:\n"
            "    pip install fairchem-core>=2.20"
        ) from exc

    import os

    if os.path.isfile(model_name):
        ckpt = model_name
    else:
        ckpt = pretrained_mlip.pretrained_checkpoint_path_from_name(model_name)

    # 'default' inference settings avoid torch.compile so autograd graphs are
    # clean for double-backprop during training.
    pu = pretrained_mlip.load_predict_unit(
        ckpt, device=device, inference_settings="default"
    )
    calc = FAIRChemCalculator(pu, task_name=task_name)
    model = pu.model.module  # trainable HydraModel
    return pu, calc, model


def uma_energy_forces(model, calc, atoms, task_name: str = "omol"):
    """Differentiable forward pass returning (energy[1], forces[N,3]) tensors.

    ``atoms`` must have ``info["charge"]`` and ``info["spin"]`` set for the
    ``omol`` head (defaults to 0 / 1 if missing).
    """
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)

    data = calc.a2g(atoms).to(next(model.parameters()).device)
    # Match the model's working dtype.
    dtype = next(model.parameters()).dtype
    for key, val in data:
        if torch.is_tensor(val) and val.is_floating_point():
            data[key] = val.to(dtype)

    out = model(data)
    energy = out[f"{task_name}_energy"]["energy"]
    forces = out[f"{task_name}_forces"]["forces"]
    return energy, forces
