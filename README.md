## VISEM-Tracking---Collective-Behavior

VISEM Tracking Dataset Link: https://zenodo.org/records/7293726

Pretrained weights for training: https://github.com/ultralytics/ultralytics

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Steps Taken for Image Preprocessing**

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

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Training the Model**

Once the dataset is prepared, we can the train the model with a pretrained weight (I have used yolos.pt)
In a notebook, run the below code, update the dataset location:

      !yolo task=detect mode=train model=yolo11s.pt data={dataset.location}/data.yaml epochs=300 imgsz=640 plots=True

Alter the parameters for your choice.
Post the train completion, results would be logged to *runs/detect/train*

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
*1. Graphs:*
Losses: Box loss, Classification loss, DFL loss.
Precision: How well the model avoids false positives.
Recall: How well the model detects true positives.
mAP50: Mean Average Precision at 50% IoU (Intersection over Union).
mAP50-95: mAP at different IoU thresholds (strict evaluation).                                                                                                                                       Interpreting results:
1. Decreasing loss → Model is improving.
2. Stable high mAP → Good detection performance.
3. High precision, low recall → Model is conservative in detection.
4. Low precision, high recall → Model detects too many false positives.

*2. Confusion Matrix*
A table that shows actual vs. predicted classifications.
True Positives (TP) → Correct detections.
False Positives (FP) → Incorrect detections (wrong class).
False Negatives (FN) → Missed detections.
Diagonal values should be high, indicating correct classifications.
Off-diagonal values indicate misclassifications.

*3. Confusion Matrix Normalized*
A version of the confusion matrix where values are scaled between 0 and 1.
Helps in comparing class-wise performance, especially when class distributions are imbalanced.
High diagonal values (close to 1) → Good classification performance.
Off-diagonal values → Areas where misclassifications occur frequently.

*4. Label Correlogram*
A visualization of how different label parameters relate.
Parameters include:

    X, Y (bounding box center coordinates).
    Width, Height (bounding box size).

High correlation values (~1.0) → Labels are highly related (possible redundancy).
Low correlation values (~0.0) → Labels are independent (good diversity in dataset).
Helps identify if features are too similar or distribution is unstructured.

