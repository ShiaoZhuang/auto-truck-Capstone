import apriltag
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO
from multiprocessing import Process, Value
import ctypes
from azure.iot.device import IoTHubDeviceClient, Message

class ArzueCommunicator:
    def __init__(self, connection_string):
        # Initialize the IoT Hub client with the provided connection string
        self.client = IoTHubDeviceClient.create_from_connection_string(connection_string)

    def send_data(self, data):
        # Convert the data to a string and wrap it in a Message object
        message = Message(str(data))
        # Send the message
        self.client.send_message(message)
        print(f"Message successfully sent: {data}")

    def __del__(self):
        # Clean up the client resources
        self.client.shutdown()
def measure_distance_continuously(shared_distance, ultrasonic_sensor):
    while True:
        distance = ultrasonic_sensor.measure_distance()
        shared_distance.value = distance
        time.sleep(0.1)  # 10 FPS

class UltrasonicSensor:
    def __init__(self, trig_pin, echo_pin):
        self.TRIG = trig_pin
        self.ECHO = echo_pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)
        GPIO.output(self.TRIG, False)
        time.sleep(2)
        print("Ultrasonic Sensor initialized successfully!")

    def measure_distance(self):
        # Ref: https://thepihut.com/blogs/raspberry-pi-tutorials/hc-sr04-ultrasonic-range-sensor-on-the-raspberry-pi
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG, False)

        pulse_start = None
        pulse_end = None

        # Wait for echo to start
        while GPIO.input(self.ECHO) == 0:
            pulse_start = time.time()
            
                
        # Wait for echo to end
        while GPIO.input(self.ECHO) == 1:
            pulse_end = time.time()

        if pulse_start is None or pulse_end is None:
            return float('inf')  # Return a very large number if measurement failed

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        return round(distance, 2)


class MotorControl:
    def __init__(self):
        # Motor GPIO setup
        print("Motors initialized successfully!")
        GPIO.setmode(GPIO.BOARD)
        self.LEFT_MOTOR_PIN = 33
        self.RIGHT_MOTOR_PIN = 32
        GPIO.setup(self.LEFT_MOTOR_PIN, GPIO.OUT)
        GPIO.setup(self.RIGHT_MOTOR_PIN, GPIO.OUT)

        self.left_motor = GPIO.PWM(self.LEFT_MOTOR_PIN, 500)
        self.right_motor = GPIO.PWM(self.RIGHT_MOTOR_PIN, 500)
        self.left_motor.start(100)
        self.right_motor.start(100)
        
    def drive(self, correction):
        # Adjust motor speeds based on PID correction
        base_speed = 10  # Base speed for forward movement
        # Scale down the correction if necessary
        correction = correction / 50
        left_speed = base_speed + correction
        right_speed = base_speed - correction

        # Clamp speeds to be within 0 to 100
        left_speed = min(max(0, left_speed), 100)
        right_speed = min(max(0, right_speed), 100)
        
        print(left_speed, right_speed)
        self.left_motor.ChangeDutyCycle(left_speed)
        self.right_motor.ChangeDutyCycle(right_speed)
        
    """
    def drive(self, direction):
        '''Maximum Speed = 0, stop = 100'''
        FORWARD_SPEED = 70 # Half speed
        TURN_SPEED = 0 # Max speed

        if direction == "Turn Right!":
            # Turn right
            self.left_motor.ChangeDutyCycle(100)
            self.right_motor.ChangeDutyCycle(TURN_SPEED)
        elif direction == "On Track!":
            # Move straight
            self.left_motor.ChangeDutyCycle(FORWARD_SPEED)
            self.right_motor.ChangeDutyCycle(FORWARD_SPEED)
        elif direction == "Turn Left!":
            # Turn left
            self.left_motor.ChangeDutyCycle(TURN_SPEED)
            self.right_motor.ChangeDutyCycle(100)
    """
            
    def stop(self, second):
        self.left_motor.ChangeDutyCycle(100)
        self.right_motor.ChangeDutyCycle(100)
        time.sleep(second)
        
    def close(self):
        self.left_motor.ChangeDutyCycle(100)
        self.right_motor.ChangeDutyCycle(100)
        self.left_motor.stop()
        self.right_motor.stop()
        
        
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
        crop_img = image[410:480, :]        # Convert to HSV
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


class Car:
    def __init__(self):
        self.camera = Camera()
        self.at_detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())
        self.motor_control = MotorControl()
        self.init_trackbar()
        self.ultrasonic_sensor = UltrasonicSensor(13, 11)  # GPIO 13 for TRIG ; GPIO 11 for ECHO
        # Shared variable for distance
        self.shared_distance = Value(ctypes.c_double, 0.0)

        # Start the distance measuring subprocess
        self.distance_process = Process(target=measure_distance_continuously, args=(self.shared_distance, self.ultrasonic_sensor))
        self.distance_process.start()
        
        self.pid_controller = PIDController(kp=20.0, ki=0.0, kd=0.01)

        self.arzue_communicator = ArzueCommunicator("HostName=DiscreetLab.azure-devices.net;DeviceId=IME2;SharedAccessKey=k93yWZQj7wg5lDLj3/ehLFRaH+unb8sLqAIoTFMNE3g=")

        self.lst_tag = None
        self.tag_id_to_city_name = {
            0: "New York",
            1: "Los Angeles",
            2: "Chicago",
            3: "Houston",
            4: "Phoenix",
            5: "Philadelphia",
            6: "San Antonio"
        }

    def init_trackbar(self):
        cv2.namedWindow('Trackbars')
        cv2.createTrackbar('Lower H', 'Trackbars', 55, 179, lambda x: None)
        cv2.createTrackbar('Lower S', 'Trackbars', 139, 255, lambda x: None)
        cv2.createTrackbar('Lower V', 'Trackbars', 74, 255, lambda x: None)
        cv2.createTrackbar('Upper H', 'Trackbars', 79, 179, lambda x: None)
        cv2.createTrackbar('Upper S', 'Trackbars', 255, 255, lambda x: None)
        cv2.createTrackbar('Upper V', 'Trackbars', 255, 255, lambda x: None)
        cv2.createTrackbar('Kp', 'Trackbars', 20, 100, lambda x: None)
        cv2.createTrackbar('Ki', 'Trackbars', 0, 100, lambda x: None)
        cv2.createTrackbar('Kd', 'Trackbars', 1, 100, lambda x: None)

    def get_trackbar_values(self):
        l_h = cv2.getTrackbarPos('Lower H', 'Trackbars')
        l_s = cv2.getTrackbarPos('Lower S', 'Trackbars')
        l_v = cv2.getTrackbarPos('Lower V', 'Trackbars')
        u_h = cv2.getTrackbarPos('Upper H', 'Trackbars')
        u_s = cv2.getTrackbarPos('Upper S', 'Trackbars')
        u_v = cv2.getTrackbarPos('Upper V', 'Trackbars')
        kp = cv2.getTrackbarPos('Kp', 'Trackbars')  # Adjusting scale if needed
        ki = cv2.getTrackbarPos('Ki', 'Trackbars') / 100 # Adjusting scale if needed
        kd = cv2.getTrackbarPos('Kd', 'Trackbars') / 100 # Adjusting scale if needed
        return np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v]), kp, ki, kd

    def run(self):
        while True:
            ''' 1st: Obstacle Avoidance'''
            distance = self.shared_distance.value
            if distance < 20:  # If distance is less than 5 cm
                print("Obstacle detected! Stopping.")
                self.motor_control.stop(1)
                continue
                
            image = self.camera.capture()
            
            ''' 2nd: Seeking for destination '''
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tags = self.at_detector.detect(gray)

            curr_id = None
            
            for tag in tags:
                # Calculate tag area as a ratio of the image area
                tag_bbox = cv2.boundingRect(np.array(tag.corners, dtype=np.int32))
                tag_area = tag_bbox[2] * tag_bbox[3] / (image.shape[0] * image.shape[1])
                
                # If the tag is half big as the image size
                if tag_area > 0.06:
                    # Draw out the AprilTag and its ID
                    for idx in range(len(tag.corners)):
                        cv2.circle(image, tuple(tag.corners[idx].astype(int)), 4, (255, 0, 0), 2)
                    tag_id_str = "id: "+str(tag.tag_id)
                    curr_id = tag.tag_id
                    position = (int(tag.center[0]), int(tag.center[1]))
                    cv2.putText(image, tag_id_str, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                    print(f"Arrived at destination {tag.tag_id}!")
                    cv2.imshow('tag', image)
                    break
            
            if curr_id != None and self.lst_tag != curr_id: # we assume the same id will not show up twice in a row
                self.lst_tag = curr_id
                time.sleep(0.5)
                self.motor_control.stop(3) # car will stop & Wait for "input" seconds
                city_name = self.tag_id_to_city_name.get(curr_id, "Unknown City")
                self.arzue_communicator.send_data(city_name)  # Send the city name to the Arzue cloud
                
                
            
            ''' 3rd: Trace the line & Move Forward '''
            # Image processing & Decision Making
            lower_hsv, upper_hsv, kp, ki, kd = self.get_trackbar_values()
            processed_image, cx, width = ImageProcessing.process_image(image, lower_hsv, upper_hsv)
            
            # calculate expected error & update PID controller
            self.pid_controller.Kp = kp
            self.pid_controller.Ki = ki
            self.pid_controller.Kd = kd

            error = (width / 2) - cx
            correction = self.pid_controller.update(error)
            
            # direction = self.decide_direction(cx, width) # simple hueristic judgement making based on threshold
            print(f"Tracing stat: {correction}")
            self.motor_control.drive(correction)
            cv2.imshow('frame', processed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   

    @staticmethod
    def decide_direction(cx, width):
        if cx >= 6/10 * width:
            return "Turn Right!"
        elif cx < 6/10 * width and cx > 4/10 * width:
            return "On Track!"
        elif cx <= 4/10 * width:
            return "Turn Left!"

class PIDController:
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error):
        # Calculate PID correction
        self.integral += error
        derivative = error - self.previous_error
        correction = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return correction


if __name__ == "__main__":
    car = Car()
    try:
        car.run()
    finally:
        car.distance_process.terminate()
        car.distance_process.join()
        car.motor_control.close()
        GPIO.cleanup() # for both motor and sensors
        cv2.destroyAllWindows()
    

    
