import cv2
from ultralytics import YOLO
import time
#import face_recognition
import numpy as np
import time

class Bone_recognize:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.fps = 0
        self.hand_up_ID = {}
        self.hand_up = False
        self.hand_up_man = None
        self.face_recognize_time = 0

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
        
        
        host_encoding = self.host(["videos/xian_1.jpg", "videos/xian_2.jpg"])
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
                                    if cv2.waitKey(1) & 0xFF == ord("c"):
                                        self.hand_up = False
                                

                                    
                                    
                                # # if 10秒:  #每十秒偵測一次                    
                                #     # 舉手辨識                                        
                                #     x1, y1, x2, y2 = (
                                #         box.xyxy[i][0].item(),
                                #         box.xyxy[i][1].item(),
                                #         box.xyxy[i][2].item(),
                                #         box.xyxy[i][3].item(),
                                #     )
                                #     face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                                    
                                    
                                #     # 人臉辨識
                                #     if self.facial_recognize(face_crop) == True:
                                #         print(f"Host position:{self.center(result, i)}",end="\n\n")
                                        # 將此人設為鎖定人
                                # elif 有鎖定的人:
                                    # print(鎖定的人的中心點)
                                    
                   
                
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Bone recognize", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "_main_":
    # demo = Bone_recognize("models/yolov8n-pose.pt", 0)
    demo = Bone_recognize("models/yolov8n-pose.pt", 0)
    demo.run()
    
    # st = time.time()
    # print(demo.facial_recognize("videos/xian_1.jpg"))
    # et = time.time()
    
    # print(et-st)
    