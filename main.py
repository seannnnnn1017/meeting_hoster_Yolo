import cv2
import time
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO('models/yolov5l6.pt')
CLASS_NAMES = model.names
# 打開視訊流
cap = cv2.VideoCapture(0)

# 初始化計數器
frame_count = 0
start_time = time.time()

# 初始化平滑過濾器
fps_filtered = 30

def fps_calulator(frame_count,fps_filtered):
    # 增加計數器
    frame_count += 1
    # 計算時間差
    elapsed_time = time.time() - start_time
    # 計算FPS
    fps = frame_count / elapsed_time
    # 應用低通濾波器
    fps_filtered = fps_filtered * 0.9 + fps * 0.1


    return fps_filtered

def detect_and_track(frame):
    results = model.track(frame, conf=0.6, save=False)

    # 获取检测结果并绘制框
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            if CLASS_NAMES[cls_id] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                # 绘制矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 在框上标记置信度
                cv2.putText(frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

while True:
    # 讀取每個影像
    ret, frame = cap.read()
    detect_and_track(frame)

    # 顯示平滑過的FPS
    cv2.putText(frame, f"FPS: {int(fps_calulator(frame_count,fps_filtered))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示影像
    cv2.imshow('Video Stream', frame)

    # 檢測按鍵輸入，按下 'esc' 停止循環
    if cv2.waitKey(1)==27:
        break

# 釋放視訊流
cap.release()

# 關閉窗口
cv2.destroyAllWindows()