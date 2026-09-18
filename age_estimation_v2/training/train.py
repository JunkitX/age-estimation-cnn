import argparse
import csv
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from age_estimation_v2.common import load_config, resolve_path, save_json, select_device, set_seed
from age_estimation_v2.data.datasets import AgeCsvDataset
from age_estimation_v2.data.split_dataset import build_split
from age_estimation_v2.models.age_model import build_model, expected_age_from_logits
from age_estimation_v2.models.losses import age_distribution_loss
from age_estimation_v2.training.evaluate import evaluate_model
from age_estimation_v2.training.utils import save_checkpoint


def build_loader(config, split, shuffle):
    dataset = AgeCsvDataset(
        split_csv=config["data"]["split_csv"],
        split=split,
        image_size=int(config["data"].get("image_size", 224)),
        min_age=int(config["data"].get("min_age", 0)),
        max_age=int(config["data"].get("max_age", 100)),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, optimizer, device, min_age, loss_config, scaler=None, grad_clip_norm=0.0):
    model.train()
    total_loss = 0.0
    total_parts = {}
    predictions = []
    targets = []
    use_amp = scaler is not None

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        ages = batch["age"].to(device)
        age_class = batch["age_class"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss, loss_parts = age_distribution_loss(
                logits,
                age_class,
                ages,
                min_age=min_age,
                **loss_config,
            )

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        pred_age, _ = expected_age_from_logits(logits.detach(), min_age=min_age)
        predictions.extend(pred_age.cpu().numpy().tolist())
        targets.extend(ages.cpu().numpy().tolist())
        total_loss += loss.item() * images.size(0)
        for name, value in loss_parts.items():
            total_parts[name] = total_parts.get(name, 0.0) + value.item() * images.size(0)

    mae = float(torch.mean(torch.abs(torch.tensor(predictions) - torch.tensor(targets))).item())
    dataset_size = len(loader.dataset)
    metrics = {
        "loss": total_loss / dataset_size,
        "mae": mae,
    }
    for name, value in total_parts.items():
        metrics[name] = value / dataset_size
    return metrics


def flatten_age_group_metrics(metrics):
    flattened = {}
    for group, values in metrics.get("age_groups", {}).items():
        key = group.replace("-", "_")
        flattened[f"val_mae_age_{key}"] = values["mae"]
        flattened[f"val_count_age_{key}"] = values["count"]
    return flattened


def append_log(path, row):
    path = resolve_path(path)
    os.makedirs(path.parent, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="age_estimation_v2/configs/train.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    split_csv = resolve_path(config["data"]["split_csv"])
    if not split_csv.exists():
        total, counts = build_split(
            resolve_path(config["data"]["data_dir"]),
            split_csv,
            seed=int(config.get("seed", 42)),
            min_age=int(config["data"].get("min_age", 0)),
            max_age=int(config["data"].get("max_age", 100)),
            stratify_age=bool(config["data"].get("stratify_age", True)),
            age_bins=[tuple(age_bin) for age_bin in config["data"].get("age_bins", [])],
        )
        print(f"Created split CSV with {total} samples: {counts}")

    set_seed(int(config.get("seed", 42)))
    device = select_device(config.get("device", "auto"))
    min_age = int(config["data"].get("min_age", 0))

    train_loader = build_loader(config, "train", shuffle=True)
    val_loader = build_loader(config, "val", shuffle=False)

    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"].get("learning_rate", 3e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["train"].get("lr_factor", 0.5)),
        patience=int(config["train"].get("lr_patience", 2)),
        min_lr=float(config["train"].get("min_learning_rate", 1e-6)),
    )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))
    loss_config = {
        "mae_weight": float(config["train"].get("mae_weight", 0.2)),
        "loss_type": config["train"].get("loss_type", "ce_l1"),
        "label_sigma": float(config["train"].get("label_sigma", 2.0)),
        "reg_weight": float(config["train"].get("reg_weight", 0.1)),
        "huber_beta": float(config["train"].get("huber_beta", 1.0)),
    }

    checkpoint_dir = Path(config["train"].get("checkpoint_dir", "age_estimation_v2/checkpoints"))
    run_dir = Path(config["train"].get("run_dir", "age_estimation_v2/runs"))
    log_path = run_dir / "train_log.csv"
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    patience = int(config["train"].get("patience", 6))

    for epoch in range(1, int(config["train"].get("epochs", 30)) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            min_age=min_age,
            loss_config=loss_config,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = evaluate_model(model, val_loader, device, min_age=min_age)
        scheduler.step(val_metrics["mae"])

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics.get("ce"),
            "train_kl_loss": train_metrics.get("kl_loss"),
            "train_age_mae_loss": train_metrics.get("mae_loss"),
            "train_huber_loss": train_metrics.get("huber_loss"),
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_within_3": val_metrics["within_3"],
            "val_within_5": val_metrics["within_5"],
            "val_within_10": val_metrics["within_10"],
        }
        row.update(flatten_age_group_metrics(val_metrics))
        append_log(log_path, row)
        print(row)

        save_checkpoint(last_path, model, optimizer, epoch, val_metrics, config)
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, epoch, val_metrics, config)
            save_json(val_metrics, run_dir / "best_val_metrics.json")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()
