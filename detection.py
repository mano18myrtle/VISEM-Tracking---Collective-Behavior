import csv
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Define paths
model_path = "/home/user/ml_env/runs/detect/Train/weights/best.pt"
input_frames_dir = "/media/user/data/Mano/Mano-Project/Project-Mano/VISEM-Tracking/VISEM_Tracking_Train_v4/Train/12/images"
output_frames_dir = "/home/user/ml_env/Detection/12/images"
csv_output_path = "/home/user/ml_env/Detection/12/12.csv"

# Create directory for output frames if it doesn't exist
os.makedirs(output_frames_dir, exist_ok=True)

# Load YOLO model
model = YOLO(model_path)

# Open CSV file for writing
with open(csv_output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header row
    writer.writerow(["Frame", "X_min", "Y_min", "X_max", "Y_max", "Confidence", "Class"])

    # Get list of image files in the input directory
    image_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    frame_number = 0  # Initialize frame counter

    for img_file in image_files:
        img_path = os.path.join(input_frames_dir, img_file)
        frame = cv2.imread(img_path)  # Read image
        
        if frame is None:
            print(f"Error loading {img_file}, skipping.")
            continue

        # Run YOLO inference on the frame
        results = model(frame, imgsz=320, conf=0.5)

        # Extract detection results
        class_counts = {0: 0, 1: 0, 2: 0}  # Initialize class counts
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confs = r.boxes.conf.cpu().numpy()  # Confidence scores
            classes = r.boxes.cls.cpu().numpy()  # Class labels
            
            for box, conf, cls in zip(boxes, confs, classes):
                x_min, y_min, x_max, y_max = map(int, box)
                class_id = int(cls)
                class_counts[class_id] += 1  # Update class count
                
                # Define class name and color
                class_names = {0: "Normal Sperm", 1: "Cluster", 2: "Pinhead Sperm"}
                class_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), class_colors[class_id], 1)  # Thinner lines

                # Display class name and confidence (Smaller font size)
                text = f"{class_names[class_id]} {conf:.2f}"
                cv2.putText(frame, text, (x_min, y_min - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, class_colors[class_id], 1)

                # Write detection results to CSV
                writer.writerow([frame_number, x_min, y_min, x_max, y_max, conf, class_id])

        
        # Display total counts on the frame (Smaller font size & Yellow color)
        count_text = f"Normal: {class_counts[0]}  Cluster: {class_counts[1]}  Pinhead: {class_counts[2]}"
        cv2.putText(frame, count_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)  # Yellow text


        # Save annotated frame
        output_frame_path = os.path.join(output_frames_dir, f"12_frame_{frame_number}.png")
        cv2.imwrite(output_frame_path, frame)

        frame_number += 1  # Increment frame count

print(f"Detections saved to {csv_output_path}")
print(f"Annotated frames saved in {output_frames_dir}")
