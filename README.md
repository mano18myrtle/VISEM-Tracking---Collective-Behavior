## VISEM-Tracking---Collective-Behavior

VISEM Tracking Dataset Link: https://zenodo.org/records/7293726

Pretrained weights for training: https://github.com/ultralytics/ultralytics

**Steps Taken for Image Preprocessing**

1. Denoising with Bilateral Filtering

   Reduces noise while preserving edges (better than Gaussian blur).
   Parameters:
        d=9 → Neighborhood diameter.
        sigmaColor=75, sigmaSpace=75 → Controls smoothing intensity.

2. Contrast Enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)

   Improves local contrast while avoiding over-enhancement.
   Parameters:
        clipLimit=0.8 → Prevents over-amplification of noise.
        tileGridSize=(4,4) → Divides the image into 4×4 regions for adaptive enhancement.

3. Edge-Preserving Sharpening
   This keeps edges crisp while making textures clearer.
   Uses a sharpening kernel to enhance fine details without excessive noise amplification.
   Kernel applied: [[0, -1, 0], 
                     [-1, 5, -1], 
                     [0, -1, 0]]


​
