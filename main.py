from ultralytics import YOLO
import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
from pynput import keyboard
import winsound

# 加載預訓練的YOLOv5模型
model = YOLO('models/yolov8n-pose.pt')
show = True
exit_program = False

def screen_capture():
    # 獲取螢幕尺寸
    screen_width = win32api.GetSystemMetrics(0)  # 獲取螢幕寬度
    screen_height = win32api.GetSystemMetrics(1)  # 獲取螢幕高度
    # 計算中心區域的座標
    left = 0
    top = 0
    # 計算縮小後的尺寸
    capture_width = screen_width
    capture_height = screen_height

    # 創建設備上下文和位圖
    hwin = win32gui.GetDesktopWindow()
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, capture_width, capture_height)
    memdc.SelectObject(bmp)
    
    # 複製屏幕中心內容到位圖
    memdc.BitBlt((0, 0), (capture_width, capture_height), srcdc, (left, top), win32con.SRCCOPY)
    
    # 將位圖轉換為numpy
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (capture_height, capture_width, 4)  # 更新形狀
    
    # 釋放資源
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    
    # 轉換為RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # 將 RGBA 轉換為 RGB
    
    return img

def draw_full_body_results(frame, result):
    if result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()
        if len(keypoints) > 0 and keypoints.shape[1] > 0:
            keypoints = keypoints[0]  # Process the first detected person only

            # Define color for keypoints and connections
            keypoint_color = (0, 255, 0)  # Green color for keypoints
            connection_color = (255, 0, 0)  # Blue color for connections
            
            # COCO body part connections
            body_connections = [
                (0,1),(0,2),(1,2),(1,3),(2,4),(1,3),(4,6),(3,5),    # head
                (6,5),(6,12),(5,11),(12,11),                        # body
                (6,8),(8,10), #right arm
                (5,7),(7,9), #left arm
                (12,14),(14,16), #right lag
                (11,13),(13,15), #left lag
            ]
            if show:
                # Draw keypoints
                for idx, (x, y) in enumerate(keypoints):
                    if x > 0 and y > 0:  # Only draw valid keypoints
                        cv2.circle(frame, (int(x), int(y)), 5, keypoint_color, -1)

                # Draw connections between keypoints
                for start_idx, end_idx in body_connections:
                    if keypoints[start_idx][0] > 0 and keypoints[end_idx][0] > 0:  # Valid points
                        start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                        end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                        cv2.line(frame, start_point, end_point, connection_color, 2)
            else:
                                # 計算骨架的右下角位置
                frame_height, frame_width, _ = frame.shape
                offset_x, offset_y = 50, 50  # 偏移量
                skeleton_position = (frame_width - offset_x, frame_height - offset_y)

                # 在米白色背景上繪製骨架
                background_color = (245, 222, 179)  # 米白色
                cv2.rectangle(frame, (skeleton_position[0] - 400, skeleton_position[1] - 400), 
                              (skeleton_position[0], skeleton_position[1]), background_color, -1)

                # 繪製骨架，將點 0 作為固定點
                for idx, (x, y) in enumerate(keypoints):
                    if x > 0 and y > 0:  # 只繪製有效的關鍵點
                        # 將所有關鍵點相對於點 0 的位置進行調整
                        adjusted_x = int(x) + skeleton_position[0] - 200 - int(keypoints[0][0])
                        adjusted_y = int(y) + skeleton_position[1] - 200 - int(keypoints[0][1])
                        cv2.circle(frame, (adjusted_x, adjusted_y), 5, keypoint_color, -1)

                # 繪製連接線
                for start_idx, end_idx in body_connections:
                    if keypoints[start_idx][0] > 0 and keypoints[end_idx][0] > 0:  # Valid points
                        start_point = (int(keypoints[start_idx][0]) + skeleton_position[0] - 200 - int(keypoints[0][0]),
                                        int(keypoints[start_idx][1]) + skeleton_position[1] - 200 - int(keypoints[0][1]))
                        end_point = (int(keypoints[end_idx][0]) + skeleton_position[0] - 200 - int(keypoints[0][0]),
                                     int(keypoints[end_idx][1]) + skeleton_position[1] - 200 - int(keypoints[0][1]))
                        cv2.line(frame, start_point, end_point, connection_color, 2)

    return frame

def on_press(key):
    global exit_program, show
    if key == keyboard.Key.end:  # 按下 End 鍵
        exit_program = True
        winsound.Beep(400, 200)
        return False  # 停止監聽
    if key == keyboard.Key.home:  # 按下 home 鍵
        show = not show
        print('Display mode changed:', 'Show' if show else 'Hide')

# 啟動鍵盤監聽
listener = keyboard.Listener(on_press=on_press)
listener.start()

# 主循環，實時捕獲和檢測
while not exit_program:
    frame = screen_capture()
    
    # 將幀轉換為模型可接受的格式
    results = model(frame, conf=0.3, iou=0.5, verbose=False)
    
    # 繪製螢幕中心點
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)  # 紅色實心圓
    
    if show:
        
        # 處理結果並畫出來
        for result in results:
            frame = draw_full_body_results(frame, result)
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.imshow("Pose Detection", frame)
    else:
        for result in results:
            frame = draw_full_body_results(frame, result)
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.imshow("Pose Detection", frame)
        

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
listener.join()
