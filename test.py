import cv2
from ultralytics import YOLO
import time
import face_recognition
import numpy as np
import time

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

class Bone_recognize:
    def __init__(self, model_path, video_path, horizontal_pin=33):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.fps = 0
        self.hand_up_ID = {}
        self.hand_up = False
        self.hand_up_man = None
        self.face_recognize = False
        self.face_recognize_time = 0
        self.face_recognize_ID = {}
        self.face_recognize_man = None
        
        
        self.motor = MotorController(horizontal_pin)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_center = [self.frame_width / 2, self.frame_height / 2]
        self.frame_counter = 0  # Counter to determine when to update the motor angle

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

    def host(self, hostphoto_path):
        known_images = hostphoto_path
        known_encodings = []

        for image_path in known_images:
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])

        # 計算平均特徵向量
        average_encoding = np.mean(known_encodings, axis=0)
        
        return average_encoding
    
    def facial_recognize(self, test_face):        
        rgb_face = cv2.cvtColor(test_face, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_face)
        unknown_encoding = face_recognition.face_encodings(rgb_face, face_locations)
        
        if len(unknown_encoding) > 0:
            unknown_encoding = unknown_encoding[0]
        else:
            return False
        
        
        host_encoding = self.host(["videos/shane.jpg"])
        results = face_recognition.compare_faces([host_encoding], unknown_encoding)
        
        if results[0]:
            return True
        else:
            return False
        
    def raise_hand(self, result, ID): 
        if result[0].keypoints.xy.numel() == 0:
            return False, False
        
        left_shoulder = result[0].keypoints.xy[ID][5]
        left_hand = result[0].keypoints.xy[ID][9]
        right_shoulder = result[0].keypoints.xy[ID][6]
        right_hand = result[0].keypoints.xy[ID][10]
        
        # print(f"left_shoulder: {left_shoulder[1]}\nright_shoulder: {right_shoulder[1]}\nleft_hand: {left_hand[1]}\nright_hand: {right_hand[1]}")

        
        left_hand_raised = False
        right_hand_raised = False
        if left_hand[1] != 0 and left_shoulder[1] != 0:
            left_hand_raised = left_hand[1] < left_shoulder[1]  
        if right_hand[1] != 0 and right_shoulder[1] != 0:
            right_hand_raised = right_hand[1] < right_shoulder[1]  

        if left_shoulder[1] == 0 or left_hand[1] == 0:
            return False, right_hand_raised
        if right_shoulder[1] == 0 or right_hand[1] == 0:
            return left_hand_raised, False

        return left_hand_raised, right_hand_raised

    def center(self, result, ID):
        box = result[0].boxes
        if ID < len(box.xyxy):
            x1, y1, x2, y2 = box.xyxy[ID][0].item(), box.xyxy[ID][1].item(), box.xyxy[ID][2].item(), box.xyxy[ID][3].item()
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            return center
        else:
            return [0, 0]

    def run(self):
        while self.cap.isOpened():
            sucess, frame = self.cap.read()
            if sucess:

                start = time.perf_counter()

                result = self.model.track(frame, conf=0.5, verbose=False)
                annotated_frame = result[0].plot()
                
                end = time.perf_counter()
                total_time = end - start
                fps = 1 / total_time


                box = result[0].boxes
                if box:
                    if box.id != None:
                        for i in range(len(box.id)):
                            name = self.model.names[int(box.cls[i].item())]
                            if name == "person":   
                                if self.hand_up == False:      
                                    left_hand_raised, right_hand_raised = self.raise_hand(result, i)
                                    if left_hand_raised == True or right_hand_raised == True:
                                        print(self.hand_up_ID)
                                        if i not in self.hand_up_ID:
                                            self.hand_up_ID[i] = 1  
                                            
                                        else:
                                            if self.hand_up_ID[i] >= 100:
                                                self.face_recognize = False
                                                self.face_recognize_ID = {}  # 清空字典
                                                self.hand_up_ID = {}  # 清空字典
                                                self.hand_up = True
                                                print(f'ID:{i + 1}')
                                                self.hand_up_man = i
                                                print(f"Hand Raised: True", end="\n\n")
                                                print(f"Hand Raised position:{self.center(result, i)}", end="\n\n")
                                                print("="*40)
                                            else:
                                                self.hand_up_ID[i] += 1
                                    else:
                                        self.hand_up_ID[i] = 0
                                            
                                elif self.hand_up == True:
                                    print(f'ID:{self.hand_up_man + 1}')
                                    print(self.center(result, self.hand_up_man))
                                    target_center = self.center(result, self.hand_up_man)
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
                                            
                                            
                                    if cv2.waitKey(1) & 0xFF == ord("c"):
                                        self.hand_up = False
                
                if self.face_recognize == False:    
                    print(self.face_recognize_time)   
                    self.face_recognize_time += 1  
                    if self.hand_up ==  False and self.face_recognize_time > 300:                                  
                        if box:
                            if box.id != None:
                                for i in range(len(box.id)):
                                    name = self.model.names[int(box.cls[i].item())]
                                    if name == "person":                                                              
                                        x1, y1, x2, y2 = (
                                            box.xyxy[i][0].item(),
                                            box.xyxy[i][1].item(),
                                            box.xyxy[i][2].item(),
                                            box.xyxy[i][3].item(),
                                        )
                                        face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                                        
                                        if self.facial_recognize(face_crop) == True:               
                                            print(self.face_recognize_ID)
                                            if i not in self.face_recognize_ID:
                                                self.face_recognize_ID[i] = 1  
                                                
                                            else:
                                                if self.face_recognize_ID[i] >= 3:
                                                    self.face_recognize_time = 0  
                                                    self.face_recognize_ID = {}  # 清空字典
                                                    self.face_recognize = True
                                                    print(f'ID:{i + 1}')
                                                    self.face_recognize_man = i
                                                    print(f"Face Recognize: True", end="\n\n")
                                                    print(f"Face Recognize position:{self.center(result, i)}", end="\n\n")
                                                    print("="*40)
                                                else:
                                                    self.face_recognize_ID[i] += 1
                                        else:
                                            self.face_recognize_ID[i] = 0
                                            
                elif self.face_recognize == True:
                    print(f'ID:{self.face_recognize_man + 1}')
                    print(self.center(result, self.face_recognize_man))
                    target_center = self.center(result, self.face_recognize_man)
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
                cv2.imshow("Bone recognize", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    demo = Bone_recognize("models/yolov8n-pose.pt", 0)
    # demo = Bone_recognize("models/yolov8n-pose.pt", "videos/xain_1.mp4")
    demo.run()
    
    # st = time.time()
    # print(demo.facial_recognize("videos/xian_1.jpg"))
    # et = time.time()
    
    # print(et-st)
    
