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
