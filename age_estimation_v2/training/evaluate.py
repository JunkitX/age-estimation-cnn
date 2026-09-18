import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from age_estimation_v2.common import load_config, save_json, select_device
from age_estimation_v2.data.datasets import AgeCsvDataset
from age_estimation_v2.common import resolve_path
from age_estimation_v2.data.split_dataset import build_split
from age_estimation_v2.models.age_model import build_model, expected_age_from_logits
from age_estimation_v2.training.metrics import age_group_metrics, regression_metrics
from age_estimation_v2.training.utils import load_checkpoint


def evaluate_model(model, loader, device, min_age=0):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            images = batch["image"].to(device)
            ages = batch["age"].to(device)
            logits = model(images)
            pred_age, _ = expected_age_from_logits(logits, min_age=min_age)
            predictions.extend(pred_age.cpu().numpy().tolist())
            targets.extend(ages.cpu().numpy().tolist())

    metrics = regression_metrics(predictions, targets)
    metrics["age_groups"] = age_group_metrics(predictions, targets)
    return metrics


def build_eval_loader(config, split):
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
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="age_estimation_v2/configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None)
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

    device = select_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device)
    loader = build_eval_loader(config, args.split)
    metrics = evaluate_model(model, loader, device, min_age=int(config["data"].get("min_age", 0)))
    print(metrics)
    if args.output:
        save_json(metrics, args.output)


if __name__ == "__main__":
    main()
