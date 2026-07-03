import os

import numpy as np
from PIL import Image


def crop_largest_face(image, margin=0.35):
    try:
        import cv2
    except ImportError:
        return image, None, "opencv_missing"

    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return image, None, "opencv_invalid"

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return image, None, "cascade_missing"

    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        return image, None, "no_face"

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad = int(max(w, h) * margin)
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image.width, x + w + pad)
    bottom = min(image.height, y + h + pad)
    return image.crop((left, top, right, bottom)), (left, top, right, bottom), "face_cropped"
