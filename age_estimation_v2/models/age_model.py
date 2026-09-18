import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights, EfficientNet_B0_Weights, ResNet18_Weights, ResNet34_Weights


class AgeDistributionModel(nn.Module):
    def __init__(self, backbone="resnet18", num_ages=101, pretrained=True, dropout=0.2):
        super().__init__()
        self.num_ages = num_ages

        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_ages))
        elif backbone == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet34(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_ages))
        elif backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            in_features = net.classifier[-1].in_features
            net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_ages))
        elif backbone == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.convnext_tiny(weights=weights)
            in_features = net.classifier[-1].in_features
            net.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_ages))
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.net = net

    def forward(self, x):
        return self.net(x)


def build_model(config):
    min_age = int(config["data"].get("min_age", 0))
    max_age = int(config["data"].get("max_age", 100))
    return AgeDistributionModel(
        backbone=config["model"].get("backbone", "resnet18"),
        num_ages=max_age - min_age + 1,
        pretrained=bool(config["model"].get("pretrained", True)),
        dropout=float(config["model"].get("dropout", 0.2)),
    )


def expected_age_from_logits(logits, min_age=0):
    probabilities = torch.softmax(logits, dim=1)
    ages = torch.arange(min_age, min_age + logits.shape[1], device=logits.device, dtype=torch.float32)
    return torch.sum(probabilities * ages, dim=1), probabilities
