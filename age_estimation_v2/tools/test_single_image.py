import argparse

from age_estimation_v2.inference.predictor import AgePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="age_estimation_v2/configs/train.yaml")
    parser.add_argument("--infer-config", default="age_estimation_v2/configs/infer.yaml")
    parser.add_argument("--no-face-crop", action="store_true")
    args = parser.parse_args()

    predictor = AgePredictor(
        args.checkpoint,
        config_path=args.config,
        infer_config_path=args.infer_config,
        crop_face=not args.no_face_crop,
    )
    result = predictor.predict_path(args.image)
    print(f"age: {result['age']:.1f}")
    print(f"confidence: {result['confidence']:.2f}")
    print(f"tta_std: {result['tta_std']:.2f}")
    print(f"entropy: {result['entropy']:.2f}")
    print(f"crop_status: {result['crop_status']}")
    print(f"face_box: {result['face_box']}")


if __name__ == "__main__":
    main()
