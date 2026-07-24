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


# ---------------------------------------------------------------------------
# LoRA support for UMA backbone scalar linear layers
# ---------------------------------------------------------------------------

import math
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing ``nn.Linear`` layer.

    The forward computes::

        base(x)  +  scaling * lora_B @ lora_A @ x

    where ``lora_A`` has shape ``(r, in_features)``, ``lora_B`` has shape
    ``(out_features, r)``, and ``scaling = alpha / r``.

    * ``base`` parameters are frozen (``requires_grad=False``).
    * ``lora_A`` is initialised with Kaiming-uniform; ``lora_B`` is zero so
      the adapter contributes nothing at initialisation (i.e. the base model
      is preserved at the start of training).
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        # Keep the frozen base layer as a sub-module so its parameters are
        # visible to state_dict() but NOT to the optimiser.
        self.base = linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero → adapter is transparent at initialisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    def extra_repr(self) -> str:
        return (f"in={self.base.in_features}, out={self.base.out_features}, "
                f"r={self.r}, scaling={self.scaling:.3f}")


def apply_lora_to_backbone(
    backbone: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_substrings: tuple = ("scalar_mlp", "rad_func.net"),
) -> int:
    """Inject :class:`LoRALinear` adapters into UMA backbone scalar linears.

    Only the invariant ``nn.Linear`` layers whose full module path contains
    any of *target_substrings* are targeted.  The equivariant
    ``SO2_Linear`` / ``SO3_Linear`` layers are untouched so rotational
    equivariance is preserved.

    Parameters
    ----------
    backbone:
        The ``model.backbone`` of a loaded UMA ``HydraModel``.
    r:
        LoRA rank.
    alpha:
        LoRA scaling numerator; effective scaling = ``alpha / r``.
    target_substrings:
        Tuple of strings.  A module is replaced iff its full dot-path name
        contains at least one of these strings.

    Returns
    -------
    Number of layers replaced.
    """
    replaced = 0
    for full_name, module in list(backbone.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(sub in full_name for sub in target_substrings):
            continue
        # Navigate to the parent container
        parts = full_name.rsplit(".", 1)
        if len(parts) == 1:
            parent, child = backbone, parts[0]
        else:
            parent = backbone
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            child = parts[1]
        setattr(parent, child, LoRALinear(module, r=r, alpha=alpha))
        replaced += 1
    return replaced

