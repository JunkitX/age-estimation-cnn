import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from age_estimation_v2.common import PROJECT_ROOT, resolve_path
from age_estimation_v2.data.transforms import build_eval_transform, build_train_transform


class AgeCsvDataset(Dataset):
    def __init__(self, split_csv, split, image_size=224, min_age=0, max_age=100):
        self.split_csv = resolve_path(split_csv)
        self.split = split
        self.min_age = min_age
        self.max_age = max_age
        self.transform = build_train_transform(image_size) if split == "train" else build_eval_transform(image_size)
        self.samples = []

        with open(self.split_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                age = int(float(row["age"]))
                if min_age <= age <= max_age:
                    image_path = Path(row["image_path"])
                    if not image_path.is_absolute():
                        image_path = PROJECT_ROOT / image_path
                    self.samples.append((image_path, age))

        if not self.samples:
            raise ValueError(f"No samples found for split '{split}' in {self.split_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, age = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        age_class = max(self.min_age, min(age, self.max_age)) - self.min_age
        return {
            "image": image,
            "age": torch.tensor(float(age), dtype=torch.float32),
            "age_class": torch.tensor(age_class, dtype=torch.long),
            "path": str(image_path),
        }
