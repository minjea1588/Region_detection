import cv2


# Open the video file
cap = cv2.VideoCapture("20240913_135821.mp4")


cnt = 0 
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    cnt += 1
    if success:
        # Run YOLOv8 inference on the frame

        # Display the annotated frame
        cv2.imwrite(f"xx2/frame_{cnt}.jpg",frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()