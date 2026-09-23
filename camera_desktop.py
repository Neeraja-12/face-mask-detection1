import sys
import os
import time
import cv2
import numpy as np
import imutils
from base_camera import BaseCamera
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Resolve paths relative to this file, so it works on Windows AND Render
BASEDIR = os.path.dirname(os.path.abspath(__file__))

PROTOTXT_PATH = os.path.join(BASEDIR, "deploy.prototxt")
WEIGHTS_PATH  = os.path.join(BASEDIR, "res10_300x300_ssd_iter_140000.caffemodel")
MODEL_PATH    = os.path.join(BASEDIR, "mask_detector.model")

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(PROTOTXT_PATH, WEIGHTS_PATH)

print("[INFO] loading face mask detector model...")
maskNet = load_model(MODEL_PATH)


class Camera(BaseCamera):
    confidence_arg = 0.5

    @staticmethod
    def detect_and_predict_mask(frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces, locs, preds = [], [], []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > Camera.confidence_arg:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                if face.size == 0:
                    continue
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)

    @staticmethod
    def frames():
        # Try to open the webcam. On Render, this will fail - fall back to
        # a placeholder image so the app does not crash.
        vs = cv2.VideoCapture(0)
        camera_available = vs.isOpened()
        if camera_available:
            time.sleep(2.0)
        else:
            print("[WARN] No camera available - serving placeholder frames.")

        while True:
            if camera_available:
                ok, frame = vs.read()
                if not ok:
                    frame = Camera._placeholder("Camera read failed")
            else:
                frame = Camera._placeholder("Camera not available on cloud server")

            frame = imutils.resize(frame, width=800)

            try:
                (locs, preds) = Camera.detect_and_predict_mask(
                    frame, faceNet, maskNet)

                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            except Exception as e:
                print("[ERROR] detection failed:", e)

            ret, jpeg = cv2.imencode('.jpg', frame)
            yield jpeg.tobytes()

    @staticmethod
    def _placeholder(text):
        img = np.zeros((480, 800, 3), dtype=np.uint8)
        cv2.putText(img, text, (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(img, "Demo runs locally with a webcam.",
                    (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return img