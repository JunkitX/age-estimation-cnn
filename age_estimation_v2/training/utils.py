import os
from pathlib import Path

import torch

from age_estimation_v2.common import resolve_path


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    path = resolve_path(path)
    os.makedirs(path.parent, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, device):
    checkpoint = torch.load(resolve_path(path), map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return checkpoint


def ensure_dir(path):
    Path(resolve_path(path)).mkdir(parents=True, exist_ok=True)

