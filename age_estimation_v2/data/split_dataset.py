import argparse
import csv
import os
import random
from pathlib import Path


def parse_age(filename, min_age=0, max_age=100):
    try:
        age = int(filename.split("_")[0])
    except (ValueError, IndexError):
        return None
    if min_age <= age <= max_age:
        return age
    return None


def build_split(data_dir, output, train_ratio=0.7, val_ratio=0.15, seed=42, min_age=0, max_age=100):
    data_dir = Path(data_dir)
    rows = []
    for file in data_dir.iterdir():
        if not file.is_file() or file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        age = parse_age(file.name, min_age=min_age, max_age=max_age)
        if age is None:
            continue
        rows.append({"image_path": str(file), "age": age})

    rng = random.Random(seed)
    rng.shuffle(rows)

    train_end = int(len(rows) * train_ratio)
    val_end = train_end + int(len(rows) * val_ratio)

    for idx, row in enumerate(rows):
        if idx < train_end:
            row["split"] = "train"
        elif idx < val_end:
            row["split"] = "val"
        else:
            row["split"] = "test"

    output = Path(output)
    os.makedirs(output.parent, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "age", "split"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), {"train": train_end, "val": val_end - train_end, "test": len(rows) - val_end}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-age", type=int, default=0)
    parser.add_argument("--max-age", type=int, default=100)
    args = parser.parse_args()

    total, counts = build_split(
        args.data_dir,
        args.output,
        seed=args.seed,
        min_age=args.min_age,
        max_age=args.max_age,
    )
    print(f"Wrote {total} samples to {args.output}: {counts}")


if __name__ == "__main__":
    main()

