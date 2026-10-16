from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11-seg.yaml")  # build a new model from YAML

    # Train the model
    results = model.train(data=r"D:\XM\XM\staw-line1.v1i.coco-segmentation\data.yaml")

