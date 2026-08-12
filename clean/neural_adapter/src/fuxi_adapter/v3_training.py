"""Training loop for the bias-anchored late-lead development experiment.

This module is deliberately separate from :mod:`fuxi_adapter.training` so the
frozen v2 workflow remains unchanged.  Each sample carries the physical fields
needed to evaluate anomaly correlation and mean bias during optimization.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .anchored import anchored_composite_loss
from .artifacts import EventLogger
from .training import set_deterministic_seed


class AnchoredSequenceDataset(Dataset):
    """In-memory sequences plus physical context for the composite loss."""

    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        bias_baseline: np.ndarray,
        truth: np.ndarray,
        climatology: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        bias_baseline = np.asarray(bias_baseline, dtype=np.float32)
        truth = np.asarray(truth, dtype=np.float32)
        climatology = np.asarray(climatology, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if features.ndim != 5:
            raise ValueError("features must have shape [case, lead, channel, lat, lon]")
        expected = (features.shape[0], features.shape[1], *features.shape[-2:])
        fields = {
            "target": target,
            "bias_baseline": bias_baseline,
            "truth": truth,
            "climatology": climatology,
            "valid_mask": valid_mask,
        }
        for name, values in fields.items():
            if values.shape != expected:
                raise ValueError(f"{name} shape {values.shape} does not match {expected}")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("features and target must be finite")
        for name, values in (
            ("bias_baseline", bias_baseline),
            ("truth", truth),
            ("climatology", climatology),
        ):
            if not np.isfinite(values[valid_mask]).all() or np.any(values[valid_mask] < 0):
                raise ValueError(f"{name} must be finite and nonnegative on valid support")
        self.features = torch.from_numpy(features)
        self.target = torch.from_numpy(target)
        self.bias_baseline = torch.from_numpy(bias_baseline)
        self.truth = torch.from_numpy(truth)
        self.climatology = torch.from_numpy(climatology)
        self.valid_mask = torch.from_numpy(valid_mask)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.features[index],
            self.target[index],
            self.bias_baseline[index],
            self.truth[index],
            self.climatology[index],
            self.valid_mask[index],
        )


@dataclass
class AnchoredTrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: pd.DataFrame
    elapsed_seconds: float


def _validate_anchor_contract(
    dataset: AnchoredSequenceDataset,
    target_scale: np.ndarray,
    *,
    name: str,
) -> None:
    """Reject a target/base/scale mismatch before optimization starts."""

    scale = np.asarray(target_scale, dtype=np.float64)
    if scale.shape != (dataset.target.shape[1],) or not np.isfinite(scale).all():
        raise ValueError("target_scale does not match the dataset lead dimension")
    if np.any(scale <= 0.0):
        raise ValueError("target_scale must be strictly positive")
    truth = dataset.truth.numpy().astype(np.float64, copy=False)
    baseline = dataset.bias_baseline.numpy().astype(np.float64, copy=False)
    target = dataset.target.numpy().astype(np.float64, copy=False)
    valid = dataset.valid_mask.numpy().astype(bool, copy=False)
    expected = np.zeros_like(target)
    expected[valid] = (
        np.log1p(truth[valid]) - np.log1p(baseline[valid])
    )
    expected /= scale[None, :, None, None]
    if not np.allclose(target[valid], expected[valid], rtol=2.0e-5, atol=2.0e-6):
        maximum = float(np.max(np.abs(target[valid] - expected[valid])))
        raise ValueError(
            f"{name} anchored target is inconsistent with baseline/scale; "
            f"maximum absolute difference={maximum:.6g}"
        )


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    area_weights: torch.Tensor,
    target_scale: torch.Tensor,
    lead_weights: torch.Tensor,
    loss_coefficients: Mapping[str, float],
    smooth_l1_beta: float,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    component_totals = {
        "smooth_l1": 0.0,
        "acc_loss": 0.0,
        "mean_spatial_acc": 0.0,
        "mean_bias_squared": 0.0,
    }
    for batch in loader:
        features, target, baseline, truth, climatology, valid_mask = (
            value.to(device, non_blocking=True) for value in batch
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda")
        with context:
            prediction = model(features)
            loss, components = anchored_composite_loss(
                prediction,
                target,
                baseline,
                truth,
                climatology,
                target_scale,
                area_weights,
                lead_weights,
                valid_mask=valid_mask,
                smooth_l1_coefficient=float(loss_coefficients["smooth_l1"]),
                acc_coefficient=float(loss_coefficients["acc"]),
                bias_coefficient=float(loss_coefficients["bias"]),
                smooth_l1_beta=smooth_l1_beta,
                return_components=True,
            )
        if training:
            assert optimizer is not None
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = int(features.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        for name, value in components.items():
            component_totals[name] += float(value.detach().cpu()) * batch_size
        samples += batch_size
    denominator = max(samples, 1)
    return total / denominator, {
        name: value / denominator for name, value in component_totals.items()
    }


def train_anchored_model(
    model: nn.Module,
    train_dataset: AnchoredSequenceDataset,
    validation_dataset: AnchoredSequenceDataset,
    area_weights: np.ndarray,
    target_scale: np.ndarray,
    lead_weights: Sequence[float],
    loss_coefficients: Mapping[str, float],
    run_directory: Path,
    *,
    seed: int,
    device: str,
    batch_size: int = 16,
    max_epochs: int = 150,
    patience: int = 20,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    smooth_l1_beta: float = 1.0,
    num_workers: int = 0,
    use_amp: bool = True,
) -> AnchoredTrainingResult:
    """Fit one candidate and select its checkpoint only by validation loss."""

    if set(loss_coefficients) != {"smooth_l1", "acc", "bias"}:
        raise ValueError("loss_coefficients must contain smooth_l1, acc, and bias")
    if not np.isclose(sum(float(value) for value in loss_coefficients.values()), 1.0):
        raise ValueError("loss coefficients must sum to one")
    _validate_anchor_contract(train_dataset, target_scale, name="train")
    _validate_anchor_contract(validation_dataset, target_scale, name="validation")
    set_deterministic_seed(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    model.to(target_device)
    area_tensor = torch.as_tensor(
        area_weights, dtype=torch.float32, device=target_device
    )
    scale_tensor = torch.as_tensor(
        target_scale, dtype=torch.float32, device=target_device
    )
    lead_tensor = torch.as_tensor(
        lead_weights, dtype=torch.float32, device=target_device
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=target_device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=target_device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1.0e-6
    )
    amp_scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp and target_device.type == "cuda"
    )
    events = EventLogger(Path(run_directory) / "logs" / "events.jsonl")
    history_rows = []
    best_loss = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale = 0
    started = time.monotonic()
    events.log(
        "anchored_training_started",
        seed=seed,
        device=str(target_device),
        train_cases=len(train_dataset),
        validation_cases=len(validation_dataset),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        lead_weights=list(float(value) for value in lead_weights),
        loss_coefficients=dict(loss_coefficients),
    )
    for epoch in range(max_epochs):
        train_loss, train_components = _epoch(
            model,
            train_loader,
            area_tensor,
            scale_tensor,
            lead_tensor,
            loss_coefficients,
            smooth_l1_beta,
            target_device,
            optimizer,
            amp_scaler,
            use_amp,
        )
        with torch.no_grad():
            validation_loss, validation_components = _epoch(
                model,
                validation_loader,
                area_tensor,
                scale_tensor,
                lead_tensor,
                loss_coefficients,
                smooth_l1_beta,
                target_device,
                None,
                None,
                use_amp,
            )
        scheduler.step(validation_loss)
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate_now,
        }
        row.update({f"train_{key}": value for key, value in train_components.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_components.items()}
        )
        history_rows.append(row)
        improved = validation_loss < best_loss - 1.0e-7
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        events.log("anchored_epoch_completed", improved=improved, **row)
        if stale >= patience:
            events.log("early_stopping", epoch=epoch, patience=patience)
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = Path(run_directory) / "checkpoints" / "best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "seed": seed,
            "target_scale": np.asarray(target_scale, dtype=np.float32),
            "lead_weights": np.asarray(lead_weights, dtype=np.float32),
            "loss_coefficients": dict(loss_coefficients),
        },
        checkpoint_path,
    )
    history = pd.DataFrame(history_rows)
    history.to_csv(Path(run_directory) / "logs" / "training_history.csv", index=False)
    elapsed = time.monotonic() - started
    events.log(
        "anchored_training_completed",
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        elapsed_seconds=elapsed,
        checkpoint=str(checkpoint_path),
    )
    return AnchoredTrainingResult(best_epoch, best_loss, history, elapsed)


__all__ = [
    "AnchoredSequenceDataset",
    "AnchoredTrainingResult",
    "train_anchored_model",
]
