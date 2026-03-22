import cv2
import time
from ultralytics import YOLO

def main():
    # 1. 加载模型 (指定使用 nano 版本兼顾速度，大赛部署时可换为您训练的 best.pt)
    model = YOLO("yolov10n.pt")
    
    # 2. 视频源设置 (测试阶段使用本地视频，现场演示时换为 RTSP 地址)
    video_source = "test_video1.mp4" 
    cap = cv2.VideoCapture(video_source)
    assert cap.isOpened(), "视频流打开失败！请检查文件路径或网络摄像头连接。"

    # 获取视频画面的宽和高
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 3. 定义虚拟撞线位置 (设为画面正中央的一条水平线)
    line_y = int(h / 2)

    # 4. 核心状态机字典与防重锁 (工业级防抖防重复计数机制)
    track_history = {}   # 记录字典：保存每个车辆ID上一帧的中心点Y坐标 {id: last_y}
    counted_ids = set()  # 黑名单锁：只要越线被数过一次的ID，永久打入黑名单，绝不重复计算

    in_count = 0         # 下行车辆计数
    out_count = 0        # 上行车辆计数
    start_time = time.time()

    print("🚀 启动边缘多模态交通感知节点...")

    # 5. 主循环：逐帧实时处理
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            print("视频流结束或中断。")
            break

        # --- A. 增强版异构多目标追踪 ---
        # classes=[2,5,7] 强制仅识别汽车、大巴、卡车
        # tracker="botsort.yaml" 启用具有抗遮挡能力的追踪器
        # imgsz=640 提高分辨率使得框更稳定，device=0 强制GPU接管算力
        results = model.track(
            frame, 
            persist=True, 
            classes=[2, 5, 7], 
            imgsz=640, 
            device=0, 
            conf=0.45, 
            tracker="botsort.yaml", 
            verbose=False
        )
        
        # --- B. 确保当前帧存在追踪目标 ---
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()         # 边界框坐标
            track_ids = results[0].boxes.id.int().cpu().numpy() # 车辆唯一ID

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2) # 当前帧该车辆的中心点 X
                cy = int((y1 + y2) / 2) # 当前帧该车辆的中心点 Y

                # 绘制检测框和中心点轨迹
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # ==========================================
                # --- C. 核心计数逻辑：基于上升沿/下降沿的严格判定 ---
                # ==========================================
                if track_id not in counted_ids: # 第一道锁：检查是否已在黑名单
                    if track_id in track_history:
                        prev_y = track_history[track_id]
                        
                        # 下降沿触发：上一帧在线上方，这一帧到了线下方 (车辆向下开)
                        if prev_y < line_y and cy >= line_y:
                            in_count += 1
                            counted_ids.add(track_id) # 永久锁死
                            
                        # 上升沿触发：上一帧在线下方，这一帧到了线上方 (车辆向上开)
                        elif prev_y > line_y and cy <= line_y:
                            out_count += 1
                            counted_ids.add(track_id) # 永久锁死

                    # 更新该车辆当前的 Y 坐标，留给下一帧做对比
                    track_history[track_id] = cy

        # --- D. 绘制虚拟警戒线 ---
        cv2.line(frame, (0, line_y), (w, line_y), (255, 0, 255), 3)

        # --- E. 边缘节点定时推流/日志打印逻辑 ---
        current_time = time.time()
        if current_time - start_time >= 5.0: # 每 5 秒输出一次日志 (可根据需求改为 60.0)
            time_str = time.strftime('%H:%M:%S')
            print(f"[{time_str}] 定时报表 -> 南向(Southbound): {in_count} 辆 | 北向(Northbound): {out_count} 辆")
            start_time = current_time

        # --- F. 本地 UI 大屏数据渲染 ---
        cv2.putText(frame, f"Southbound: {in_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.putText(frame, f"Northbound: {out_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        cv2.imshow("Industrial AI Traffic Monitor", frame)
        
        # 按键盘 'q' 键安全退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放显存与视频句柄资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()