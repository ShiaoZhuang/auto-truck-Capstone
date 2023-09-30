import pupil_apriltags as apriltag
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

at_detector = apriltag.Detector(families='tag25h9')

recording = False
video_writer = None

while (1):
    # Get frame
    ret, frame = cap.read()
    # Check key presses
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('s'):
        if recording:
            # Stop recording and release video writer
            video_writer.release()
            video_writer = None
        else:
            # Start recording
            height, width, layers = frame.shape
            video_name = '/Users/zhuang/Downloads/QR-code/video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to write video
            video_writer = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))
        recording = not recording  # Toggle recording flag

    # Detect apriltag
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray)
    for tag in tags:
        cv2.circle(frame, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2)  # left-top
        cv2.circle(frame, tuple(tag.corners[1].astype(int)), 4, (255, 0, 0), 2)  # right-top
        cv2.circle(frame, tuple(tag.corners[2].astype(int)), 4, (255, 0, 0), 2)  # right-bottom
        cv2.circle(frame, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2)  # left-bottom

        # Draw tag ID on the video
        tag_id_str = "id: "+str(tag.tag_id)
        position = (int(tag.center[0]), int(tag.center[1]))
        cv2.putText(frame, tag_id_str, position, cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2, cv2.LINE_AA)

    # If recording, write frame to video file
    if recording and video_writer is not None:
        video_writer.write(frame)

    # Show detection results
    cv2.imshow('capture', frame)

if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()
