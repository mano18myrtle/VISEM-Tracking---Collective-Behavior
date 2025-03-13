import os
import cv2
import numpy as np
import csv
from ultralytics import YOLO

# ========== CONFIGURATION ==========
video_path = "/media/user/data/Mano/Mano-Project/Project-Mano/VISEM-Tracking/VISEM_Tracking_Train_v4/Train/11/11.mp4"
model_path = "/home/user/ml_env/runs/detect/Train/weights/best.pt"
output_counts_csv = "/home/user/ml_env/Detection_validation/11_detection.csv"

# Class names dictionary for readable labels
class_names = {0: "Normal Sperm", 1: "Cluster", 2: "Pinhead Sperm"}

def count_sperm_cells(video_path, model_path, output_csv):
    """Count different types of sperm cells in each frame of a video and save to CSV"""
    # Load YOLO model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize data collection
    sperm_counts_data = []
    frame_count = 0
    
    print(f"Starting sperm cell counting for {video_path}...")
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Initialize counts for this frame
        current_frame_counts = {"sperm_count": 0, "cluster_count": 0, "small_or_pinhead_count": 0}
        
        # Run model detection
        results = model(frame, verbose=False)[0]
        
        # Count detections by class
        for i, box in enumerate(results.boxes.xyxy):
            class_id = int(results.boxes.cls[i])
            
            if class_id == 0:  # Normal Sperm
                current_frame_counts["sperm_count"] += 1
            elif class_id == 1:  # Cluster
                current_frame_counts["cluster_count"] += 1
            elif class_id == 2:  # Pinhead Sperm
                current_frame_counts["small_or_pinhead_count"] += 1
        
        # Add counts to data
        frame_name = f"11_frame_{frame_count}"
        sperm_counts_data.append({
            "frame_name": frame_name,
            "sperm_count": current_frame_counts["sperm_count"],
            "cluster_count": current_frame_counts["cluster_count"],
            "small_or_pinhead_count": current_frame_counts["small_or_pinhead_count"]
        })
        
        # Update progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
        
        frame_count += 1
    
    # Release video
    cap.release()
    
    # Save data to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['frame_name', 'sperm_count', 'cluster_count', 'small_or_pinhead_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in sperm_counts_data:
            writer.writerow(data)
    
    print(f"Counting completed! Processed {frame_count} frames.")
    print(f"Sperm count data saved to {output_csv}")

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_counts_csv), exist_ok=True)
    
    # Run the counting function
    count_sperm_cells(video_path, model_path, output_counts_csv)

