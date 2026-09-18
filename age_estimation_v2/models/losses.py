import torch
import torch.nn.functional as F

from age_estimation_v2.models.age_model import expected_age_from_logits


def gaussian_age_distribution(true_age, num_ages, min_age=0, sigma=2.0):
    ages = torch.arange(min_age, min_age + num_ages, device=true_age.device, dtype=torch.float32)
    distances = ages.unsqueeze(0) - true_age.unsqueeze(1)
    distribution = torch.exp(-0.5 * (distances / sigma) ** 2)
    return distribution / distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)


def age_distribution_loss(
    logits,
    age_class,
    true_age,
    min_age=0,
    mae_weight=0.2,
    loss_type="ce_l1",
    label_sigma=2.0,
    reg_weight=0.1,
    huber_beta=1.0,
):
    if loss_type == "ce_l1":
        return ce_l1_loss(logits, age_class, true_age, min_age=min_age, mae_weight=mae_weight)
    if loss_type == "gaussian_kl_huber":
        return gaussian_kl_huber_loss(
            logits,
            true_age,
            min_age=min_age,
            label_sigma=label_sigma,
            reg_weight=reg_weight,
            huber_beta=huber_beta,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def ce_l1_loss(logits, age_class, true_age, min_age=0, mae_weight=0.2):
    ce = F.cross_entropy(logits, age_class)
    expected_age, _ = expected_age_from_logits(logits, min_age=min_age)
    mae = F.l1_loss(expected_age, true_age)
    return ce + mae_weight * mae, {"ce": ce.detach(), "mae_loss": mae.detach()}


def gaussian_kl_huber_loss(logits, true_age, min_age=0, label_sigma=2.0, reg_weight=0.1, huber_beta=1.0):
    target_distribution = gaussian_age_distribution(
        true_age=true_age,
        num_ages=logits.shape[1],
        min_age=min_age,
        sigma=label_sigma,
    )
    log_probabilities = F.log_softmax(logits, dim=1)
    kl = F.kl_div(log_probabilities, target_distribution, reduction="batchmean")
    expected_age, _ = expected_age_from_logits(logits, min_age=min_age)
    huber = F.smooth_l1_loss(expected_age, true_age, beta=huber_beta)
    return kl + reg_weight * huber, {"kl_loss": kl.detach(), "huber_loss": huber.detach()}
