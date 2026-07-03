import json
import os
import random
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path):
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        # The bundled configs are YAML, but also valid enough to be converted manually only
        # for simple debugging is not worth supporting. Install PyYAML for normal use.
        raise RuntimeError("PyYAML is required to read config files. Install with: pip install PyYAML")


def select_device(name="auto"):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data, path):
    path = resolve_path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
