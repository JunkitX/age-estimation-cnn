from pathlib import Path

import torch
from PIL import Image

from age_estimation_v2.common import load_config, select_device
from age_estimation_v2.data.face_crop import crop_largest_face
from age_estimation_v2.data.transforms import build_eval_transform
from age_estimation_v2.models.age_model import build_model, expected_age_from_logits
from age_estimation_v2.training.utils import load_checkpoint


class AgePredictor:
    def __init__(
        self,
        checkpoint=None,
        config_path="age_estimation_v2/configs/train.yaml",
        infer_config_path=None,
        device="auto",
        crop_face=True,
    ):
        self.config = load_config(config_path)
        self.infer_config = load_config(infer_config_path) if infer_config_path else {}
        checkpoint = checkpoint or self.infer_config.get("checkpoint")
        if checkpoint is None:
            raise ValueError("A checkpoint path is required.")

        if device == "auto":
            device = self.infer_config.get("device", self.config.get("device", "auto"))
        self.device = select_device(device)
        data_config = self.config["data"]
        self.min_age = int(self.infer_config.get("min_age", data_config.get("min_age", 0)))
        self.max_age = int(self.infer_config.get("max_age", data_config.get("max_age", 100)))
        self.face_margin = float(self.infer_config.get("face_margin", 0.35))
        self.crop_face = crop_face
        image_size = int(self.infer_config.get("image_size", data_config.get("image_size", 224)))
        self.transform = build_eval_transform(image_size)
        self.model = build_model(self.config).to(self.device)
        self.checkpoint = load_checkpoint(checkpoint, self.model, self.device)
        self.model.eval()

    def predict_image(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        face_box = None
        crop_status = "disabled"
        face_image = image
        if self.crop_face:
            face_image, face_box, crop_status = crop_largest_face(image, margin=self.face_margin)

        tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        flipped = self.transform(face_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)).unsqueeze(0).to(self.device)
        batch = torch.cat([tensor, flipped], dim=0)

        with torch.no_grad():
            logits = self.model(batch)
            expected_ages, probabilities = expected_age_from_logits(logits, min_age=self.min_age)
            mean_probability = probabilities.mean(dim=0)
            age = expected_ages.mean().item()
            tta_std = expected_ages.std(unbiased=False).item()
            entropy = -torch.sum(mean_probability * torch.log(mean_probability + 1e-10)).item()
            top_probability = mean_probability.max().item()

        confidence = max(0.0, min(1.0, top_probability * 8.0))
        return {
            "age": age,
            "tta_std": tta_std,
            "entropy": entropy,
            "confidence": confidence,
            "face_box": face_box,
            "crop_status": crop_status,
            "face_image": face_image,
        }

    def predict_path(self, image_path):
        return self.predict_image(Path(image_path))
