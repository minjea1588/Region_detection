import cv2
from Region_detection import Manages_Parts

# Open the video file
cap = cv2.VideoCapture("20240913_140012.mp4")

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
cnt = 0
json_path = "bounding_boxes.json"
management = Manages_Parts(model_path = "best_0905_v10_cpcm_L.pt")
video_writer = cv2.VideoWriter("parking management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
# Loop through the video frames


while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        json_data = management.parking_regions_extraction(json_path)
        results = management.model.track(frame, persist=True, show=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            management.process_data(json_data, frame, boxes, clss)

        management.display_frames(frame)
        video_writer.write(frame)
        # Break the loop if 'q' is pressed

# Release the video capture object and close the display window
cap.release()
video_writer.release()
cv2.destroyAllWindows()