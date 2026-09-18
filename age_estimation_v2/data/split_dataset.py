import argparse
import csv
import os
import random
from pathlib import Path


DEFAULT_AGE_BINS = [(0, 12), (13, 19), (20, 35), (36, 50), (51, 100)]


def parse_age(filename, min_age=0, max_age=100):
    try:
        age = int(filename.split("_")[0])
    except (ValueError, IndexError):
        return None
    if min_age <= age <= max_age:
        return age
    return None


def parse_age_bins(value):
    bins = []
    for part in value.split(","):
        low, high = part.split("-")
        bins.append((int(low), int(high)))
    return bins


def split_counts(total, train_ratio, val_ratio):
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if total >= 3 and val_ratio > 0 and val_count == 0:
        val_count = 1
        if test_count > train_count and test_count > 1:
            test_count -= 1
        else:
            train_count = max(1, train_count - 1)

    if total >= 2 and test_count == 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        else:
            val_count = max(0, val_count - 1)

    return train_count, val_count, test_count


def assign_split(rows, train_ratio, val_ratio, rng):
    rng.shuffle(rows)
    train_count, val_count, _ = split_counts(len(rows), train_ratio, val_ratio)
    train_end = train_count
    val_end = train_count + val_count

    counts = {"train": 0, "val": 0, "test": 0}
    for idx, row in enumerate(rows):
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"
        row["split"] = split
        counts[split] += 1
    return counts


def assign_stratified_split(rows, train_ratio, val_ratio, rng, age_bins):
    counts = {"train": 0, "val": 0, "test": 0}
    grouped = {age_bin: [] for age_bin in age_bins}
    grouped["other"] = []

    for row in rows:
        age = row["age"]
        for low, high in age_bins:
            if low <= age <= high:
                grouped[(low, high)].append(row)
                break
        else:
            grouped["other"].append(row)

    for group_rows in grouped.values():
        if not group_rows:
            continue
        group_counts = assign_split(group_rows, train_ratio, val_ratio, rng)
        for split, count in group_counts.items():
            counts[split] += count

    rng.shuffle(rows)
    return counts


def build_split(
    data_dir,
    output,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    min_age=0,
    max_age=100,
    stratify_age=False,
    age_bins=None,
):
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
    if stratify_age:
        counts = assign_stratified_split(rows, train_ratio, val_ratio, rng, age_bins or DEFAULT_AGE_BINS)
    else:
        counts = assign_split(rows, train_ratio, val_ratio, rng)

    output = Path(output)
    os.makedirs(output.parent, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "age", "split"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-age", type=int, default=0)
    parser.add_argument("--max-age", type=int, default=100)
    parser.add_argument("--stratify-age", action="store_true")
    parser.add_argument("--age-bins", default="0-12,13-19,20-35,36-50,51-100")
    args = parser.parse_args()

    total, counts = build_split(
        args.data_dir,
        args.output,
        seed=args.seed,
        min_age=args.min_age,
        max_age=args.max_age,
        stratify_age=args.stratify_age,
        age_bins=parse_age_bins(args.age_bins),
    )
    print(f"Wrote {total} samples to {args.output}: {counts}")


if __name__ == "__main__":
    main()
