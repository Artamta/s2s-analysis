"""Minimal deterministic PyTorch training loop for the residual adapters."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import EventLogger


class SequenceDataset(Dataset):
    """In-memory forecast sequences; the complete dataset is only tens of MiB."""

    def __init__(self, features: np.ndarray, target: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if features.ndim != 5:
            raise ValueError("features must have shape [case, lead, channel, lat, lon]")
        if target.shape != (features.shape[0], features.shape[1], *features.shape[-2:]):
            raise ValueError("target shape does not match feature cases/leads/grid")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("features and target must be finite after preprocessing")
        self.features = torch.from_numpy(features)
        self.target = torch.from_numpy(target)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.target[index]


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def masked_weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spatial_weights: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Area-weighted Smooth-L1 mean over cases, leads, and valid India cells."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must match [batch, lead, lat, lon]")
    if spatial_weights.shape != prediction.shape[-2:]:
        raise ValueError("spatial_weights do not match prediction grid")
    valid_weights = torch.where(
        torch.isfinite(spatial_weights) & (spatial_weights > 0),
        spatial_weights,
        torch.zeros_like(spatial_weights),
    )
    if torch.sum(valid_weights) <= 0:
        raise ValueError("spatial_weights contain no positive support")
    element = torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction="none", beta=beta
    )
    weights = valid_weights[None, None]
    return torch.sum(element * weights) / (
        torch.sum(valid_weights) * prediction.shape[0] * prediction.shape[1]
    )


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: pd.DataFrame
    elapsed_seconds: float


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    weights: torch.Tensor,
    beta: float,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda")
        with context:
            prediction = model(features)
            loss = masked_weighted_smooth_l1(prediction, target, weights, beta)
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
        batch = features.shape[0]
        total += float(loss.detach().cpu()) * batch
        samples += batch
    return total / max(samples, 1)


def train_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    spatial_weights: np.ndarray,
    run_directory: Path,
    *,
    seed: int,
    device: str,
    batch_size: int = 16,
    max_epochs: int = 150,
    patience: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    beta: float = 1.0,
    num_workers: int = 0,
    use_amp: bool = True,
) -> TrainingResult:
    """Fit a model, choose its checkpoint only from validation loss, and log all epochs."""

    set_deterministic_seed(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    model.to(target_device)
    weights = torch.as_tensor(spatial_weights, dtype=torch.float32, device=target_device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        SequenceDataset(train_features, train_target),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=target_device.type == "cuda",
    )
    validation_loader = DataLoader(
        SequenceDataset(validation_features, validation_target),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=target_device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
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
        "training_started",
        seed=seed,
        device=str(target_device),
        train_cases=len(train_features),
        validation_cases=len(validation_features),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
    )
    for epoch in range(max_epochs):
        train_loss = _epoch(
            model, train_loader, weights, beta, target_device, optimizer, amp_scaler, use_amp
        )
        with torch.no_grad():
            validation_loss = _epoch(
                model,
                validation_loader,
                weights,
                beta,
                target_device,
                None,
                None,
                use_amp,
            )
        scheduler.step(validation_loss)
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate_now,
            }
        )
        improved = validation_loss < best_loss - 1e-7
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        events.log(
            "epoch_completed",
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate_now,
            improved=improved,
        )
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
        },
        checkpoint_path,
    )
    history = pd.DataFrame(history_rows)
    history.to_csv(Path(run_directory) / "logs" / "training_history.csv", index=False)
    elapsed = time.monotonic() - started
    events.log(
        "training_completed",
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        elapsed_seconds=elapsed,
        checkpoint=str(checkpoint_path),
    )
    return TrainingResult(best_epoch, best_loss, history, elapsed)


def predict(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: str,
    batch_size: int = 32,
    use_amp: bool = True,
) -> np.ndarray:
    """Predict standardized residuals without targets or shuffled ordering."""

    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 5 or not np.isfinite(feature_array).all():
        raise ValueError("features must be finite [case, lead, channel, lat, lon]")
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(feature_array), batch_size):
            batch = torch.from_numpy(feature_array[start : start + batch_size]).to(
                target_device
            )
            with torch.cuda.amp.autocast(
                enabled=use_amp and target_device.type == "cuda"
            ):
                outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)

