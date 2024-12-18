import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time
import numpy as np

class MotorController:
    def __init__(self, horizontal_pin):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(horizontal_pin, GPIO.OUT)

        # Initialize PWM with 50Hz frequency
        self.horizontal_servo = GPIO.PWM(horizontal_pin, 50)
        
        # Start the PWM signal with 0 duty cycle to prevent sudden movement
        self.horizontal_servo.start(0)

        # Set the horizontal servo to 90 degrees (initial position)
        time.sleep(1)  # Allow time for the servo to initialize
        self.set_angle('horizontal', 90)

        # Wait a little to ensure the servo reaches the desired position
        time.sleep(2)
        
        self.horizontal_angle = 90

    def set_angle(self, axis, angle):
        if axis == 'horizontal':
            self.horizontal_angle = max(50, min(130, angle))  # Angle between 50 and 130 degrees
            duty_cycle = 2.5 + (0.05 * self.horizontal_angle)  # Servo angle calculation
            self.horizontal_servo.ChangeDutyCycle(duty_cycle)

    def cleanup(self):
        self.horizontal_servo.stop()
        GPIO.cleanup()

class BoneRecognizeWithoutPID:
    def __init__(self, model_path, video_path, horizontal_pin=33):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.motor = MotorController(horizontal_pin)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_center = [self.frame_width / 2, self.frame_height / 2]
        self.frame_counter = 0  # Counter to determine when to update the motor angle

    def center(self, result, ID):
        box = result[0].boxes
        if ID < len(box.xyxy):
            x1, y1, x2, y2 = box.xyxy[ID][0].item(), box.xyxy[ID][1].item(), box.xyxy[ID][2].item(), box.xyxy[ID][3].item()
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            return center
        else:
            return [0, 0]

    def track_object(self, target_center):
        # Calculate the difference between the target center and the frame center
        diff_x = target_center[0] - self.frame_center[0]

        # Adjust the horizontal angle based on the center difference
        horizontal_adjustment = diff_x / 20  # Fine-tuning this factor

        # Apply the adjustment to the motor angle
        new_horizontal_angle = self.motor.horizontal_angle - horizontal_adjustment

        # Set the new horizontal motor angle
        self.motor.set_angle('horizontal', new_horizontal_angle)

        return new_horizontal_angle

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                start = time.perf_counter()
                result = self.model.track(frame, conf=0.5, verbose=False)
                annotated_frame = result[0].plot()
                end = time.perf_counter()
                fps = 1 / (end - start)

                box = result[0].boxes
                if box and len(box.xyxy) > 0:
                    for i in range(len(box.xyxy)):
                        name = self.model.names[int(box.cls[i].item())]
                        if name == "person":
                            target_center = self.center(result, i)
                            if target_center != [0, 0]:
                                # Update motor angle only every 10 frames
                                if self.frame_counter % 10 == 0:
                                    new_horizontal_angle = self.track_object(target_center)
                                    diff_x = target_center[0] - self.frame_center[0]
                                    print(f"Tracking ID: {i + 1}")
                                    print(f"Center Difference: X={diff_x}")
                                    print(f"Motor Angle - Horizontal: {new_horizontal_angle:.2f}")

                                # Increment the frame counter
                                self.frame_counter += 1
                                if self.frame_counter >= 10:
                                    self.frame_counter = 0

                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Bone recognize without PID", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.motor.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = BoneRecognizeWithoutPID("models/yolov8n-pose.pt", 0)
    demo.run()
