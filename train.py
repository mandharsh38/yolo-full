from ultralytics import YOLO
import os
import cv2

def train_model():
    model = YOLO('yolov8m-seg.pt')
    model.train(
        data="data.yaml",
        epochs=500,
        imgsz=640,
        patience=50,
        batch=64,
#        resume=True,
	device=0
    )

if __name__ == "__main__":
    train_model()
