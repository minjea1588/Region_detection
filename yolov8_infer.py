from ultralytics import YOLO

model = YOLO("0916_best_L_v8.pt")
results = model("CPCM_side.jpg", save = True)