import apriltag
import cv2
import numpy as np
from picamera2 import Picamera2
import time

class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}, controls={"ExposureTime":5000}))
        self.picam2.start()
        
    def capture(self):
        return self.picam2.capture_array()
    
    def stop(self):
        self.picam2.stop()

class ImageProcessing:
    @staticmethod
    def process_image(image, lower_hsv, upper_hsv):
        # Crop the image
        crop_img = image[0:640, :]
        # Convert to HSV
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        # Create a mask based on the lower and upper HSV values
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        # Apply the mask to the original image
        res = cv2.bitwise_and(crop_img, crop_img, mask=mask)
        # Find the centroid of the line
        m = cv2.moments(mask, False)
        height, width = mask.shape
        try:
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
        except ZeroDivisionError:
            cx, cy = width/2, height/2
        # Draw a line from the centroid to the top and bottom of the image
        cv2.line(res, (int(cx), 0), (int(cx), height), (255, 0, 0), 1)
        # Draw a line from the centroid to the left and right of the image
        cv2.line(res, (0, int(cy)), (width, int(cy)), (255, 0, 0), 1)
        # Return the result and the x-coordinate of the centroid
        return res, cx, width

class MotorControl:
    def drive(self, direction):
        # This method would contain the logic to control the motors.
        # For now, it just prints the direction.
        print(direction)

class Car:
    def __init__(self):
        self.camera = Camera()
        self.at_detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())
        self.motor_control = MotorControl()
        self.init_trackbar()

    def init_trackbar(self):
        cv2.namedWindow('Trackbars')
        cv2.createTrackbar('Lower H', 'Trackbars', 0, 179, lambda x: None)
        cv2.createTrackbar('Lower S', 'Trackbars', 0, 255, lambda x: None)
        cv2.createTrackbar('Lower V', 'Trackbars', 0, 255, lambda x: None)
        cv2.createTrackbar('Upper H', 'Trackbars', 179, 179, lambda x: None)
        cv2.createTrackbar('Upper S', 'Trackbars', 255, 255, lambda x: None)
        cv2.createTrackbar('Upper V', 'Trackbars', 255, 255, lambda x: None)

    def get_trackbar_values(self):
        l_h = cv2.getTrackbarPos('Lower H', 'Trackbars')
        l_s = cv2.getTrackbarPos('Lower S', 'Trackbars')
        l_v = cv2.getTrackbarPos('Lower V', 'Trackbars')
        u_h = cv2.getTrackbarPos('Upper H', 'Trackbars')
        u_s = cv2.getTrackbarPos('Upper S', 'Trackbars')
        u_v = cv2.getTrackbarPos('Upper V', 'Trackbars')
        return np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])

    def run(self):
        while True:
            image = self.camera.capture()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tags = self.at_detector.detect(gray)
            tag_flag = 0
            for tag in tags:
                # Calculate tag area as a ratio of the image area
                tag_bbox = cv2.boundingRect(np.array(tag.corners, dtype=np.int32))
                tag_area = tag_bbox[2] * tag_bbox[3] / (image.shape[0] * image.shape[1])
                
                # If the tag is half big as the image size
                if tag_area > 0.5:
                    # Draw out the AprilTag and its ID
                    for idx in range(len(tag.corners)):
                        cv2.circle(image, tuple(tag.corners[idx].astype(int)), 4, (255, 0, 0), 2)
                    tag_id_str = "id: "+str(tag.tag_id)
                    position = (int(tag.center[0]), int(tag.center[1]))
                    cv2.putText(image, tag_id_str, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                    print(f"Arrived at destination {tag.tag_id}!")
                    cv2.imshow('tag', image)
                    tag_flag = 1
                    #time.sleep(5)  # Wait for 5 seconds
                    break
                    
            # if arrive destination
            if tag_flag:
                # TODO
                continue
                
            lower_hsv, upper_hsv = self.get_trackbar_values()
            processed_image, cx, width = ImageProcessing.process_image(image, lower_hsv, upper_hsv)
            direction = self.decide_direction(cx, width)
            self.motor_control.drive(direction)
            cv2.imshow('frame', processed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    @staticmethod
    def decide_direction(cx, width):
        if cx >= 7/10 * width:
            return "Turn Right!"
        elif cx < 7/10 * width and cx > 3/10 * width:
            return "On Track!"
        elif cx <= 3/10 * width:
            return "Turn Left"

if __name__ == "__main__":
    car = Car()
    car.run()
    cv2.destroyAllWindows()
