## VISEM-Tracking---Collective-Behavior

VISEM Tracking Dataset Link: https://zenodo.org/records/7293726

Pretrained weights for training: https://github.com/ultralytics/ultralytics

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Steps Taken for Image Preprocessing**

Note: Refer Preprocessing_script.ipynb and Batch_Processing.ipynb

1. Denoising with Bilateral Filtering

   Reduces noise while preserving edges (better than Gaussian blur).

         Parameters:

         d=9 → Neighborhood diameter.

         sigmaColor=75, sigmaSpace=75 → Controls smoothing intensity.

3. Contrast Enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)

   Improves local contrast while avoiding over-enhancement.

         Parameters:

         clipLimit=0.8 → Prevents over-amplification of noise.
   
         tileGridSize=(4,4) → Divides the image into 4×4 regions for adaptive enhancement.

5. Edge-Preserving Sharpening

   This keeps edges crisp while making textures clearer.

   Uses a sharpening kernel to enhance fine details without excessive noise amplification.

         Kernel applied:
   
                     [[0, -1, 0], 
                     [-1, 5, -1], 
                      [0, -1, 0]]

**Other Preprocessing methods which were performed**

Note: Setting_up_and_Classic_methods.ipynb, Classic methods.ipynb and Diff Preprocessing.ipynb

1. Background Subtraction:

   Remove uneven backgrounds caused by illumination variation

   Use polynomial fitting methods or grayscale morphology operations
   
2. Thresholding:

   Identifies object on an image/removes noise

   Adaptive thresholding-segment sperm heads from background
   
3. Opening-Closing Steps:

    Apply morphological opening to remove small objects and smooth boundaries

    Closing- To fill small holes and connect nearby objects
   
4. Erosion/Dilation:
   
   Erosion - To shrinnk objects and seperate connected components

   Dilation - To grow objects and fill small gaps
   
5. Contrast Enhancement:

   Improves image contrast, to highlight sperm structures
   
6. Noise Reduction:

   Filtering techniques to remove artifacts

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ML on VISEM Dataset
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Dataset Preparation for Model Training**

Prepare a data yaml file in the format:

      # Define the number of classes
      nc: 3 

      # Define the number of classes
      names: ["Normal Sperm", "Cluster", "Pinhead Sperm"] 
      
      # Paths to dataset directories
      train: /path/to/train/images  # Path to training images
      val: /path/to/val/images  # Path to validation images
      test: /path/to/test/images  # (Optional) Path to test images

Structure the dataset following the data yaml file: 

      /dataset_root/
      │── train/
      │   ├── images/
      │   │   ├── img1.jpg
      │   │   ├── img2.jpg
      │   │   ├── ...
      │   ├── labels/
      │   │   ├── img1.txt
      │   │   ├── img2.txt
      │   │   ├── ...
      │
      │── val/
      │   ├── images/
      │   ├── labels/
      │
      │── test/  # (Optional)
      │   ├── images/
      │   ├── labels/
      │
      │── data.yaml  # The YAML configuration file

Note: Refer VISEM_Model_Training.ipynb for the Model Training using YOLOv11 

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Training the Model**

Once the dataset is prepared, we can the train the model with a pretrained weight (I have used yolos.pt)
Download the appropriate pretrained weight for your model.
In a notebook, run the below code, update the dataset location:

      !yolo task=detect mode=train model=yolo11s.pt data={dataset.location}/data.yaml epochs=300 imgsz=640 plots=True

Alter the parameters for your choice.
Post the train completion, trained weights and the results would be logged to *runs/detect/train*

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Validating the Model**

      !yolo task=detect mode=val \
          model=/home/user/ml_env/runs/detect/Train/weights/best.pt \
          data={dataset.location}/data.yaml \
          imgsz=640 \
          batch=16 \
          conf=0.25 \
          iou=0.6 \
          plots=True \
          save_txt \
          save_conf \
          name=Validate
Results would be stored in *runs/detect/val*

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Testing the Model**

      !yolo task=detect mode=predict \
          model=/home/user/ml_env/runs/detect/Train/weights/best.pt \
          conf=0.25 \
          source={dataset.location}/test/images \
          save=True \
          save_txt=True \
          save_conf=True \
          project=/home/user/ml_env/runs/detect \
          name=Test \
          exist_ok=True
Results would be stored in */home/user/ml_env/runs/detect/Test*

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
With above three steps, we can evaluate the performace of the training with the help of the performance metrics

*1. Graphs*

Losses: Box loss, Classification loss, DFL loss. Decreasing loss → Model is improving.

Precision: How well the model avoids false positives.

Recall: How well the model detects true positives.

mAP50: Mean Average Precision at 50% IoU (Intersection over Union). 

mAP50-95: mAP at different IoU thresholds (strict evaluation). Stable high mAP → Good detection performance. 

High precision, low recall → Model is conservative in detection.

Low precision, high recall → Model detects too many false positives.

*2. Confusion Matrix*

A table that shows actual vs. predicted classifications.

True Positives (TP) → Correct detections.

False Positives (FP) → Incorrect detections (wrong class).

False Negatives (FN) → Missed detections.

Diagonal values should be high, indicating correct classifications. Off-diagonal values indicate misclassifications.

*3. Confusion Matrix Normalized*

A version of the confusion matrix where values are scaled between 0 and 1. Helps in comparing class-wise performance, especially when class distributions are imbalanced.

High diagonal values (close to 1) → Good classification performance.

Off-diagonal values → Areas where misclassifications occur frequently.

*4. Label Correlogram*

A visualization of how different label parameters relate.

Parameters include:

    X, Y (bounding box center coordinates)
    Width, Height (bounding box size)

High correlation values (~1.0) → Labels are highly related (possible redundancy).

Low correlation values (~0.0) → Labels are independent (good diversity in dataset).

*5. F1 Curve*

A graph of F1-score vs Confidence Threshold.

      F1-score = 2 × (Precision × Recall) / (Precision + Recall)

The curve helps find the optimal confidence threshold for detection.

Sharp drops in the curve → Model’s confidence needs better calibration.

Peak F1-score indicates the best balance between precision and recall.

Helps identify if features are too similar or distribution is unstructured.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Sperm Detection and Validation
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*1. Detection*

Note: Refer detection.py

Detect sperm cells in image frames using the trained model. Draw bounding boxes around detected objects. Save detection results (bounding box coordinates, confidence scores, class labels) in a CSV file. Generate and save annotated frames with labeled sperm detections.

Workflow:

 Loads the trained model.
 Reads images from the input directory.
 Runs YOLO inference to detect:
 
      Normal Sperm
      Cluster
      Pinhead Sperm
Saves:
   Detection results in a CSV file (frame, x_min, y_min, x_max, y_max, confidence, class_id).
   Annotated images in the output directory.
   Displays total counts per frame on the image.

Output:

   Annotated images in /Detection/<video_id>/images/
               
   ![12_frame_73](https://github.com/user-attachments/assets/bf45223e-d2d0-4ea8-94e1-7ce7ba748e2a)   
                                                        
   12_frame_73

   Detection results in /Detection/<video_id>/<video_id>.csv

*2. Validation*

Note: Refer validation.py

Validate the detection performance on a video file. Count the number of detected sperm cells frame by frame. Log detection counts into a CSV file for further analysis.

Workflow:

Loads the trained model.
Opens the input video file.
Processes each frame:

      Runs YOLO inference.
      Counts occurrences of each sperm type.
      Logs frame-wise counts (frame_name, sperm_count, cluster_count, small_or_pinhead_count).
Saves the detection count summary to a CSV file.

Output:

   Frame-wise sperm count summary in /Detection_validation/<video_id>_detection.csv

--------------------------------------------------------------------------------------------------------------------------------------------------------------
Sperm Tracking 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Requirements

    Python
    OpenCV
    NumPy
    Pandas
    SciPy
    Ultralytics YOLO
    FilterPy (for Kalman filter)
    Openpyxl (for Excel logging)

*1. Video Processing & Initialization*

Before processing frames, we need to load the video, YOLO model, and initialize tracking structures.

Code Snippet: Load Video & YOLO Model

      import cv2
      import numpy as np
      from ultralytics import YOLO
      from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm

      # Load YOLO Model (Ensure the model file is in the directory)
      model = YOLO("yolov8s.pt")  # Replace with the trained sperm detection model

      # Open Video File
      cap = cv2.VideoCapture("sperm_video.mp4")

      # Initialize tracking dictionaries
      trackers = {}  # Stores sperm objects
      next_id = 0  # Unique ID counter

*2. Object Detection (Using YOLO)*

Each frame is passed through the YOLO model to detect sperm. The model returns bounding boxes, centroids, and class labels.

Code Snippet: YOLO Detection

      def detect_sperm(frame):
          detections = model(frame)[0]  # YOLO prediction
          boxes = detections.boxes.xyxy.cpu().numpy()  # Bounding boxes
          confidences = detections.boxes.conf.cpu().numpy()  # Confidence scores
          class_labels = detections.boxes.cls.cpu().numpy()  # Class labels

          sperm_data = []
          for box, conf, label in zip(boxes, confidences, class_labels):
              x1, y1, x2, y2 = box  # Bounding box coordinates
              cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # Compute centroid
              sperm_data.append([cx, cy, x1, y1, x2, y2, conf, int(label)])  # Store data

          return sperm_data

*3. Tracking Sperm Across Frames*

Tracking is handled using Kalman filters to predict sperm positions and Hungarian algorithm for ID matching.

Step 1: Predict Next Position Using Kalman Filter

We initialize a Kalman filter for each detected sperm to predict its next position.

Code Snippet: Kalman Filter Setup

      import cv2

      def create_kalman_filter():
          kf = cv2.KalmanFilter(4, 2)
          kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], np.float32)
          kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32)
          kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
          return kf

Step 2: Associate Detections with Existing Tracks

The Hungarian Algorithm is used to associate new detections with existing sperm tracks based on IoU (Intersection over Union).

Code Snippet: IoU Calculation

      def compute_iou(box1, box2):
          x1, y1, x2, y2 = box1
          x1p, y1p, x2p, y2p = box2

          xi1 = max(x1, x1p)
          yi1 = max(y1, y1p)
          xi2 = min(x2, x2p)
          yi2 = min(y2, y2p)
          inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

          box1_area = (x2 - x1) * (y2 - y1)
          box2_area = (x2p - x1p) * (y2p - y1p)
          iou = inter_area / (box1_area + box2_area - inter_area)
    
          return iou

Step 3: Track Sperm Using Hungarian Algorithm

We assign unique IDs to sperm and update their positions across frames.

Code Snippet: Hungarian Algorithm for ID Assignment

      def assign_ids(sperm_data, trackers):
          global next_id

          if len(trackers) == 0:
              for sperm in sperm_data:
                  kf = create_kalman_filter()
                  kf.statePre[:2] = np.array([[sperm[0]], [sperm[1]]], np.float32)
                  trackers[next_id] = {"kf": kf, "box": sperm, "age": 0}
                  next_id += 1
          else:
              existing_ids = list(trackers.keys())
              existing_boxes = [trackers[tid]["box"] for tid in existing_ids]
              new_boxes = [sperm[:4] for sperm in sperm_data]

              if existing_boxes and new_boxes:
                  cost_matrix = np.zeros((len(existing_boxes), len(new_boxes)))
                  for i, old_box in enumerate(existing_boxes):
                      for j, new_box in enumerate(new_boxes):
                          cost_matrix[i, j] = -compute_iou(old_box, new_box)

                  row_ind, col_ind = linear_sum_assignment(cost_matrix)

                  for r, c in zip(row_ind, col_ind):
                     trackers[existing_ids[r]]["box"] = sperm_data[c]

*4. Visualization*

We draw bounding boxes and trajectories to visualize sperm movement.

Code Snippet: Drawing on Frame

      def draw_tracks(frame, trackers):
          for tid, data in trackers.items():
              x1, y1, x2, y2 = data["box"][:4]
              cx, cy = data["box"][0], data["box"][1]

              # Draw bounding box
              cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

              # Draw ID label
              cv2.putText(frame, f"ID: {tid}", (int(cx), int(cy - 10)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

*5. Data Logging*

Tracking data is saved into an Excel file for further analysis.

Code Snippet: Save Data to Excel

      import pandas as pd

      tracking_data = []

      def save_tracking_data(trackers, frame_number):
          for tid, data in trackers.items():
              cx, cy = data["box"][0], data["box"][1]
              tracking_data.append({"Frame": frame_number, "ID": tid, "X": cx, "Y": cy})

      df = pd.DataFrame(tracking_data)
      df.to_csv("sperm_tracking_data.csv", index=False)

*6. Frame Capture & Saving*

Save frames periodically for analysis.

Code Snippet: Save Frames

      frame_count = 0
      while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
              break

          sperm_data = detect_sperm(frame)
          assign_ids(sperm_data, trackers)
          draw_tracks(frame, trackers)

          if frame_count % 50 == 0:  # Save every 50 frames
              cv2.imwrite(f"frames/frame_{frame_count}.png", frame)

          save_tracking_data(trackers, frame_count)

          frame_count += 1

Final Outcome

The system detects sperm using YOLO, tracks movement with Kalman filter, assigns IDs using Hungarian algorithm, and logs tracking data.
Output includes:

1. Tracking visualization with bounding boxes and IDs.
![12_frame_0-100](https://github.com/user-attachments/assets/c6b0e359-fd0c-49d3-966d-1cedefb74082)

   12_frame_0-100

3. CSV file with sperm trajectories.

4. Saved frames for further analysis.
   
