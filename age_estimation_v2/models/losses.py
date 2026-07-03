import torch.nn.functional as F

from age_estimation_v2.models.age_model import expected_age_from_logits


def age_distribution_loss(logits, age_class, true_age, min_age=0, mae_weight=0.2):
    ce = F.cross_entropy(logits, age_class)
    expected_age, _ = expected_age_from_logits(logits, min_age=min_age)
    mae = F.l1_loss(expected_age, true_age)
    return ce + mae_weight * mae, {"ce": ce.detach(), "mae_loss": mae.detach()}

