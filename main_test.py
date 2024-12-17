import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time
import face_recognition 
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(50, 130)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits

        self.previous_error = 0
        self.integral = 0

    def compute(self, setpoint, measured_value):
        error = setpoint - measured_value
        self.integral += error
        derivative = error - self.previous_error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = max(min(output, self.output_limits[1]), self.output_limits[0])  # 限制輸出範圍

        self.previous_error = error
        return output

class MotorController:
    def __init__(self, horizontal_pin, vertical_pin):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(horizontal_pin, GPIO.OUT)
        GPIO.setup(vertical_pin, GPIO.OUT)

        self.horizontal_servo = GPIO.PWM(horizontal_pin, 50)  # PWM 頻率 50Hz
        self.vertical_servo = GPIO.PWM(vertical_pin, 50)

        self.horizontal_servo.start(7.5)  # 初始位置
        self.vertical_servo.start(7.5)

        self.horizontal_angle = 90
        self.vertical_angle = 90

    def set_angle(self, axis, angle):
        if axis == 'horizontal':
            self.horizontal_angle = max(50, min(130, angle))
            duty_cycle = 2.5 + (self.horizontal_angle / 18)
            self.horizontal_servo.ChangeDutyCycle(duty_cycle)
        elif axis == 'vertical':
            self.vertical_angle = max(50, min(130, angle))
            duty_cycle = 2.5 + (self.vertical_angle / 18)
            self.vertical_servo.ChangeDutyCycle(duty_cycle)

    def cleanup(self):
        self.horizontal_servo.stop()
        self.vertical_servo.stop()
        GPIO.cleanup()

class BoneRecognizeWithPID:
    def __init__(self, model_path, video_path, horizontal_pin=32, vertical_pin=33):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.motor = MotorController(horizontal_pin, vertical_pin)
        self.pid_horizontal = PIDController(0.1, 0.01, 0.05)
        self.pid_vertical = PIDController(0.1, 0.01, 0.05)

        self.frame_center = [640 / 2, 480 / 2]  # 假設解析度為 640x480
        self.hand_up_ID = {}
        self.hand_up = False
        self.hand_up_man = None

    def center(self, result, ID):
        box = result[0].boxes
        if ID < len(box.xyxy):
            x1, y1, x2, y2 = box.xyxy[ID][0].item(), box.xyxy[ID][1].item(), box.xyxy[ID][2].item(), box.xyxy[ID][3].item()
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            return center
        else:
            return [0, 0]

    def track_object(self, target_center):
        # 計算水平方向和垂直方向的 PID 輸出
        horizontal_adjustment = self.pid_horizontal.compute(self.frame_center[0], target_center[0])
        vertical_adjustment = self.pid_vertical.compute(self.frame_center[1], target_center[1])

        # 設定馬達角度
        new_horizontal_angle = self.motor.horizontal_angle + horizontal_adjustment
        new_vertical_angle = self.motor.vertical_angle + vertical_adjustment

        self.motor.set_angle('horizontal', new_horizontal_angle)
        self.motor.set_angle('vertical', new_vertical_angle)

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
                                self.track_object(target_center)
                                if not self.hand_up:
                                    print(f"Tracking ID: {i + 1}")
                                    print(f"Position: {target_center}")

                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Bone recognize with PID", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.motor.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = BoneRecognizeWithPID("models/yolov8n-pose.pt", 0)
    demo.run()
