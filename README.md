# auto-truck-Capstone
Overall: used to store the 2023 Fall Capstone Project's software portion.
## Beta Prototype: Line Tracing & AprilTag Detection

### Features
**Line Tracing:** The robot follows a line using computer vision. It processes live footage from a camera to identify the line and calculates the necessary steering angles to stay on course.
**AprilTag Detection:** While following the line, the robot continuously scans for AprilTags in its field of view. Upon detecting an AprilTag, it calculates the tag's area in relation to the image area.
**Arrival Logic:** If a detected AprilTag's area is more than half of the image area, the robot considers it has reached its destination. It then executes a predefined action (like stopping or performing a specific maneuver).

### How It Works
The program operates in a continuous loop, performing the following steps:

1. Capture an image from the camera.
2. Process the image to detect the line and calculate its position.
3. Scan the processed image for AprilTags.
4. If an AprilTag is detected and its area is more than half of the image area, interpret this as the destination.
5. If at the destination, perform the destination action (like stopping for 5 seconds). Otherwise, continue line tracing.
6. Calculate the steering angle based on the line's position and send the corresponding commands to the motor controller.
7. Repeat.

-----------------

## Alpha Prototype: AprilTag Detection and Video Recording
This project provides a script for detecting AprilTags in a live video stream, and allows the user to start and stop video recording with a keypress.

### Requirements
Ensure you have the required libraries installed using the following command:
```bash
pip install -r requirements.txt
```

### Usage
Run the script using your preferred Python IDE or from the command line with the following command:
```bash
python aprilTag_detect.py
```


- Once the script is running, the live video feed from your webcam will display in a window titled "capture".

- Press the 's' key to start recording. The video will record and display the AprilTag detections in real-time. The ID of each detected AprilTag will be displayed on the video feed.

- Press the 's' key again to stop recording. The recorded video will be saved as video.mp4 in the specified directory.

- Press the 'Esc' key to exit the script and close the video window.

### Implementation
The script utilizes the pupil-apriltags library for AprilTag detection and the cv2 library (OpenCV) for handling video capture, recording, and display.

- The cv2.VideoCapture class is used to capture the live video feed from the webcam.
- The pupil-apriltags library is used to detect any AprilTags in each frame of the video.
- The cv2.VideoWriter class is used to handle video recording.
- The cv2.putText method is used to overlay the ID of each detected AprilTag on the video feed.
- The cv2.imshow method is used to display the live video feed, and the cv2.waitKey method is used to check for keypresses to control video recording and exit the script.
- The script runs in a continuous loop, processing and displaying each frame of the video feed until the user exits the script with the 'Esc' key.
