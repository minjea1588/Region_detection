import cv2
from Region_detection import Manages_Parts


cap = cv2.VideoCapture("test_video.mp4")

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
cnt = 0
json_path = "frame_5_bounding_boxes.json"
management = Manages_Parts(model_path = "0916_cpcm_S_KFold_v8.pt")
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
    else:
        break

# Release the video capture object and close the display window
cap.release()
video_writer.release()
cv2.destroyAllWindows()