import argparse
from tkinter import Button, Label, Tk, filedialog

from PIL import ImageTk

from age_estimation_v2.inference.predictor import AgePredictor


class AgeEstimatorApp:
    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor
        self.root.title("Age Estimation V2")
        self.root.geometry("760x720")

        self.image_label = Label(root)
        self.image_label.pack(pady=16)

        self.result_label = Label(root, text="请选择一张人脸照片", font=("Microsoft YaHei", 12), justify="left")
        self.result_label.pack(pady=12)

        Button(
            root,
            text="选择图片",
            command=self.open_image,
            font=("Microsoft YaHei", 11),
            padx=18,
            pady=10,
        ).pack(pady=16)

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        try:
            result = self.predictor.predict_path(path)
            preview = ImageTk.PhotoImage(result["face_image"].resize((360, 360)))
            self.image_label.config(image=preview)
            self.image_label.image = preview

            face_status = {
                "face_cropped": "已检测并裁剪最大人脸",
                "no_face": "未检测到人脸，使用整图预测",
                "opencv_missing": "未安装 OpenCV，使用整图预测",
                "opencv_invalid": "OpenCV 安装不完整，使用整图预测",
                "cascade_missing": "OpenCV 人脸检测模型缺失，使用整图预测",
                "disabled": "已关闭人脸裁剪",
            }.get(result["crop_status"], result["crop_status"])

            text = (
                f"预测年龄：{result['age']:.1f} 岁\n"
                f"置信提示：{result['confidence']:.2f}\n"
                f"增强波动：±{result['tta_std']:.1f} 岁\n"
                f"分布熵：{result['entropy']:.2f}\n"
                f"预处理：{face_status}"
            )
            self.result_label.config(text=text)
        except Exception as exc:
            self.result_label.config(text=f"预测失败：{exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="age_estimation_v2/checkpoints/best_model.pth")
    parser.add_argument("--config", default="age_estimation_v2/configs/train.yaml")
    parser.add_argument("--no-face-crop", action="store_true")
    args = parser.parse_args()

    predictor = AgePredictor(args.checkpoint, config_path=args.config, crop_face=not args.no_face_crop)
    root = Tk()
    app = AgeEstimatorApp(root, predictor)
    root.mainloop()


if __name__ == "__main__":
    main()
