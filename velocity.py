import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import random
import openpyxl
from openpyxl import Workbook
import colorsys

# ========== CONFIGURATION ==========
model_path = "/home/user/Mano/ml_env/runs/detect/Train/weights/best.pt"
output_base_dir = "/home/user/Mano/ml_env/Velocity/"

# Class names dictionary for readable labels
class_names = {0: "Normal Sperm", 1: "Cluster", 2: "Pinhead Sperm"}

max_distance = 50  # Distance threshold for ID association
frame_interval = 100  # Save frame every 100 frames
max_missed_frames = 10  # Remove track if missed for too long
display = False  # Toggle real-time display

# ========== COLOR GENERATION ==========
def generate_distinct_colors(n):
    distinct_colors = []
    golden_ratio_conjugate = 0.618033988749895
    fixed_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 255), (255, 0, 128)
    ]
    distinct_colors.extend(fixed_colors[:min(len(fixed_colors), n)])
    if n > len(fixed_colors):
        h = random.random()
        for i in range(n - len(fixed_colors)):
            h += golden_ratio_conjugate
            h %= 1
            r, g, b = colorsys.hsv_to_rgb(h, 0.99, 0.99)
            bgr = (int(b * 255), int(g * 255), int(r * 255))
            if bgr not in distinct_colors:
                distinct_colors.append(bgr)
    random.shuffle(distinct_colors)
    return distinct_colors

# ========== TRACKER CLASS ==========
class Track:
    def __init__(self, track_id, centroid, class_name, bbox=None):
        self.track_id = track_id
        self.centroids = [centroid]
        self.missed_frames = 0
        self.active = True
        self.class_name = class_name
        self.times = [0]
        self.velocities = [(0, 0)]
        self.bboxes = [bbox] if bbox is not None else None
        
        self.kalman = KalmanFilter(dim_x=4, dim_z=2)
        self.kalman.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kalman.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kalman.P *= 500
        self.kalman.R = np.eye(2) * 0.05
        self.kalman.Q = np.eye(4) * 0.05
        self.kalman.x[:2] = np.array([[centroid[0]], [centroid[1]]])

    def predict(self):
        self.kalman.predict()
        pred = self.kalman.x[:2].flatten()
        return int(pred[0]), int(pred[1])

    def update(self, centroid, frame_count, fps, bbox=None):
        predicted_pos = self.predict()
        smoothed_x = int(0.8 * centroid[0] + 0.2 * predicted_pos[0])
        smoothed_y = int(0.8 * centroid[1] + 0.2 * predicted_pos[1])
        self.kalman.update([smoothed_x, smoothed_y])
        
        current_time = frame_count / fps
        self.times.append(current_time)
        
        if len(self.centroids) > 0:
            prev_x, prev_y = self.centroids[-1]
            time_diff = current_time - self.times[-2]
            if time_diff > 0:
                vx = (smoothed_x - prev_x) / time_diff
                vy = (smoothed_y - prev_y) / time_diff
                self.velocities.append((vx, vy))
            else:
                self.velocities.append((0, 0))
        
        self.centroids.append((smoothed_x, smoothed_y))
        if bbox is not None and self.bboxes is not None:
            self.bboxes.append(bbox)
        
        self.missed_frames = 0
        self.active = True

# Function to safely save a frame
def safe_imwrite(filepath, img):
    if img is not None and img.size > 0:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return cv2.imwrite(filepath, img)
    return False

# ========== IMPROVED TRACKLET ASSOCIATION ==========
def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0
    return intersection / union

# ========== PROCESS VIDEO FUNCTION ==========
def process_video(video_id):
    video_path = f"/media/user/data/Mano/Mano-Project/VISEM-Tracking/VISEM_Tracking_Train_v4/Train/{video_id}/{video_id}.mp4"
    output_video = f"{output_base_dir}{video_id}/{video_id}.mp4"
    output_xlsx = f"{output_base_dir}{video_id}/{video_id}.xlsx"
    output_frames_dir = f"{output_base_dir}{video_id}/tracking_frames"

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    tracking_data = []
    tracks = {}
    next_track_id = 0
    frame_count = 0

    estimated_max_tracks = 50
    distinct_colors = generate_distinct_colors(estimated_max_tracks)
    track_colors = {}

    captured_frames = set()
    last_valid_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        last_valid_frame = frame.copy()
        frame_count += 1
        results = model(frame, verbose=False)[0]
        
        detections = []
        for i, box in enumerate(results.boxes.xyxy):
            bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
            class_id = int(results.boxes.cls[i])
            class_name = class_names.get(class_id, "Unknown")
            confidence = float(results.boxes.conf[i]) if hasattr(results.boxes, 'conf') else 1.0
            detections.append((centroid, class_name, bbox, confidence))

        detections.sort(key=lambda x: x[3], reverse=True)

        predicted = {track_id: (track.predict(), track) for track_id, track in tracks.items()}
        matched_tracks = set()

        if predicted and detections:
            cost_matrix = np.zeros((len(predicted), len(detections)))
            pred_keys = list(predicted.keys())
            
            for i, track_id in enumerate(pred_keys):
                pred_center, track = predicted[track_id]
                for j, (det_center, det_class, det_bbox, _) in enumerate(detections):
                    position_cost = np.linalg.norm(np.array(pred_center) - np.array(det_center))
                    class_cost = 0 if track.class_name == det_class else 20
                    iou_cost = 0
                    if track.bboxes and len(track.bboxes) > 0:
                        iou = calc_iou(track.bboxes[-1], det_bbox)
                        iou_cost = (1 - iou) * 10
                    cost_matrix[i, j] = position_cost + class_cost + iou_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < max_distance + 20:
                    track_id = pred_keys[r]
                    det_center, det_class, det_bbox, _ = detections[c]
                    tracks[track_id].update(det_center, frame_count, fps, det_bbox)
                    matched_tracks.add(track_id)
                    tracking_data.append([
                        frame_count, 
                        track_id, 
                        det_center[0], 
                        det_center[1], 
                        tracks[track_id].velocities[-1][0], 
                        tracks[track_id].velocities[-1][1], 
                        det_class
                    ])

            for i, (det_center, det_class, det_bbox, _) in enumerate(detections):
                if i not in col_ind:
                    tracks[next_track_id] = Track(next_track_id, det_center, det_class, det_bbox)
                    track_colors[next_track_id] = distinct_colors[next_track_id % len(distinct_colors)]
                    tracking_data.append([frame_count, next_track_id, det_center[0], det_center[1], 0, 0, det_class])
                    next_track_id += 1

        elif detections:
            for det_center, det_class, det_bbox, _ in detections:
                tracks[next_track_id] = Track(next_track_id, det_center, det_class, det_bbox)
                track_colors[next_track_id] = distinct_colors[next_track_id % len(distinct_colors)]
                tracking_data.append([frame_count, next_track_id, det_center[0], det_center[1], 0, 0, det_class])
                next_track_id += 1

        for track_id in list(tracks.keys()):
            if track_id not in matched_tracks:
                tracks[track_id].missed_frames += 1
                if tracks[track_id].missed_frames > max_missed_frames:
                    del tracks[track_id]

        for track in tracks.values():
            color = track_colors.get(track.track_id, (0, 255, 0))
            for i in range(1, len(track.centroids)):
                cv2.line(frame, track.centroids[i - 1], track.centroids[i], color, 1)
            label = f'id:{track.track_id} class:{track.class_name}'
            cv2.putText(frame, label, track.centroids[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        out.write(frame)

        if (frame_count % frame_interval == 0 or frame_count == 1) and frame is not None:
            start_range = ((frame_count - 1) // frame_interval) * frame_interval
            end_range = start_range + frame_interval
            filename = f"{video_id}_frame_{start_range}-{end_range}.png"
            safe_imwrite(os.path.join(output_frames_dir, filename), frame)
            captured_frames.add(start_range)

    if last_valid_frame is not None:
        last_start_range = (frame_count // frame_interval) * frame_interval
        if last_start_range not in captured_frames:
            filename = f"{video_id}_frame_{last_start_range}-{frame_count}.png"
            safe_imwrite(os.path.join(output_frames_dir, filename), last_valid_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    wb = Workbook()
    wb.remove(wb.active)
    df = pd.DataFrame(tracking_data, columns=["Frame", "ID", "X", "Y", "V_x", "V_y", "Class"])
    track_data = {}
    for data in tracking_data:
        track_id = data[1]
        if track_id not in track_data:
            track_data[track_id] = []
        track_data[track_id].append(data)

    for track_id, data_list in track_data.items():
        class_name = data_list[0][6]
        tab_name = f"ID_{track_id}_{class_name}"[:31]
        try:
            ws = wb.create_sheet(title=tab_name)
            ws.append(["Time", "X", "Y", "V_x", "V_y"])
            for data in data_list:
                frame_num, _, x, y, vx, vy, _ = data
                time_sec = frame_num / fps
                ws.append([time_sec, x, y, vx, vy])
        except Exception as e:
            print(f"Error creating sheet {tab_name}: {e}")

    try:
        wb.save(output_xlsx)
    except Exception as e:
        print(f"Error saving Excel file: {e}")

    print(f"Processing completed for video {video_id}")

# ========== BATCH PROCESSING ==========
video_ids = [11, 12, 13, 14, 15, 19, 21, 22, 23, 24, 29, 30, 35, 36, 38, 47, 52, 54, 60, 82]

for video_id in video_ids:
    process_video(str(video_id))

print("Batch processing completed for all videos!")

