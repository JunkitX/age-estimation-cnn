from pathlib import Path

import torch
from PIL import Image

from age_estimation_v2.common import load_config, select_device
from age_estimation_v2.data.face_crop import crop_largest_face
from age_estimation_v2.data.transforms import build_eval_transform
from age_estimation_v2.models.age_model import build_model, expected_age_from_logits
from age_estimation_v2.training.utils import load_checkpoint


class AgePredictor:
    def __init__(self, checkpoint, config_path="age_estimation_v2/configs/train.yaml", device="auto", crop_face=True):
        self.config = load_config(config_path)
        if device == "auto":
            device = self.config.get("device", "auto")
        self.device = select_device(device)
        self.min_age = int(self.config["data"].get("min_age", 0))
        self.max_age = int(self.config["data"].get("max_age", 100))
        self.crop_face = crop_face
        self.transform = build_eval_transform(int(self.config["data"].get("image_size", 224)))
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
            face_image, face_box, crop_status = crop_largest_face(image)

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

