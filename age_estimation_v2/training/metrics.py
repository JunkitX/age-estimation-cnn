import numpy as np


def regression_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    errors = np.abs(predictions - targets)
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "within_3": float(np.mean(errors <= 3.0)),
        "within_5": float(np.mean(errors <= 5.0)),
        "within_10": float(np.mean(errors <= 10.0)),
    }


def age_group_metrics(predictions, targets):
    groups = [(0, 12), (13, 19), (20, 35), (36, 50), (51, 100)]
    output = {}
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    for low, high in groups:
        mask = (targets >= low) & (targets <= high)
        key = f"{low}-{high}"
        if not np.any(mask):
            output[key] = {"count": 0, "mae": None}
        else:
            output[key] = {
                "count": int(np.sum(mask)),
                "mae": float(np.mean(np.abs(predictions[mask] - targets[mask]))),
            }
    return output

