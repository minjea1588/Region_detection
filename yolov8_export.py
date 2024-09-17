from ultralytics import YOLO
# Load a model
model = YOLO("yolov8l.pt")  # load an official model
model = YOLO("best_0905_v10_cpcm_L.pt")  # load a custom trained model

# Export the model
model.export(format="openvino", half=True)