from ultralytics import YOLO


def main():
    model = YOLO('yolov8x-seg.yaml').load('yolov8x-seg.pt')  # build from YAML and transfer weights
    model.train(data=r'D:\code\miccai\2d\coco128-seg.yaml', epochs=3000, imgsz=1024, batch=2, workers=0,patience=500)


if __name__ == '__main__':
    main()