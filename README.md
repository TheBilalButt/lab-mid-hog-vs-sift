# Computer Vision Lab Mid Exam: HOG vs SIFT on Blood-Cell Images

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheBilalButt/lab-mid-hog-vs-sift/blob/main/42_BilalButt_LabMid.ipynb)
![Submitted to Dr. Jamal Hussain Shah](https://img.shields.io/badge/Submitted%20to-Dr.%20Jamal%20Hussain%20Shah-0f766e.svg?style=for-the-badge)
![Student Bilal Butt](https://img.shields.io/badge/Student-Bilal%20Butt%20(Roll%2042)-1d4ed8.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed%20%26%20Verified-10b981.svg?style=for-the-badge)

**Course:** Computer Vision Lab (CS-312L) | **Exam:** Midterm Examination  
**Instructor:** **Dr. Jamal Hussain Shah**  
**Student Name:** Muhammad Bilal Butt  
**Registration Number:** FA23-BAI-042 (Roll No: **42**)  
**Degree:** BS Artificial Intelligence  
**Google Colab Notebook:** [Open in Google Colab](https://colab.research.google.com/github/TheBilalButt/lab-mid-hog-vs-sift/blob/main/42_BilalButt_LabMid.ipynb)  
**GitHub Profile:** [@TheBilalButt](https://github.com/TheBilalButt)  

---

## Executive Summary & Personal Allocation (Reg No: 42)

This repository contains the verified implementation and comparative analysis of two classical computer vision pipelines—**HOG + LinearSVC** and **Dense SIFT + LinearSVC**—evaluated on the **BloodMNIST** biomedical benchmark dataset. The goal is to determine classification reliability under environmental distortions (**+20° rotation** and **additive Gaussian noise $\sigma = 0.05$**).

### Parameter Settings for Registration Number 42:
- **Last Two Digits:** `42` $\rightarrow$ **Random Seed = 42**
- **Last Digit:** `2` (Even) $\rightarrow$ **HOG Cell Size = 8 x 8**
- **Last Digit:** `2` (0 to 4) $\rightarrow$ **SIFT Grid Size = 4 x 4 regular grid (16 keypoints, size 32)**
- **Classes:** 4 blood cell classes: eosinophil (1), monocyte (5), neutrophil (6), platelet (7)
- **Sample Distribution:** 100 train, 30 validation, 30 test images per class (Total: 400 train, 120 val, 120 test)
- **Preprocessing:** Single-channel grayscale conversion, resized to $128 \times 128$
- **Classifier:** `LinearSVC`, tuning $C \in \{0.1, 1.0, 10.0\}$ on validation set

---

## Deliverables in this Repository
- [`42_BilalButt_LabMid.ipynb`](42_BilalButt_LabMid.ipynb) — Executed Jupyter Notebook with all outputs pre-rendered.
- [`42_BilalButt_LabMid.pdf`](42_BilalButt_LabMid.pdf) — High-resolution PDF submission document.
- [`README.md`](README.md) — Complete markdown report with all visible outputs and calculations.
- [`assets/`](assets/) — All output figures and visualization charts.
- [`requirements.txt`](requirements.txt) — Environment dependencies.

---

## Q1. Data Setup (3 marks)

### Extracted Shapes & Class Balance:
```text
=== Split Shapes ===
X_train shape: (400, 128, 128), y_train shape: (400,)
X_val shape:   (120, 128, 128), y_val shape:   (120,)
X_test shape:  (120, 128, 128), y_test shape:  (120,)

=== Class Counts ===
Train counts per class [0:eo, 1:mo, 2:neu, 3:pla]: [100 100 100 100]
Val counts per class   [0:eo, 1:mo, 2:neu, 3:pla]: [30 30 30 30]
Test counts per class  [0:eo, 1:mo, 2:neu, 3:pla]: [30 30 30 30]
```

### Preprocessed Sample Grid (4 Examples per Class):
![Q1 Sample Grid](assets/q1_sample_grid.png)

---

## Q2. HOG + SVM Pipeline (6 marks)

### Hand Calculation of HOG Feature Vector Length:
For Registration Number **42** (cell size **$8 \times 8$**):
1. **Grid Cells:** $\text{Cells horizontally} = \frac{128}{8} = 16$; $\text{Cells vertically} = \frac{128}{8} = 16 \implies 16 \times 16 = 256 \text{ cells}$.
2. **Overlapping Blocks ($2 \times 2$ cells, stride 1):**
   $$\text{Horizontal blocks} = (16 - 2) + 1 = 15$$
   $$\text{Vertical blocks} = (16 - 2) + 1 = 15$$
   $$\text{Total Blocks} = 15 \times 15 = 225 \text{ blocks}$$
3. **Features per Block:** $2 \times 2 \times 9 \text{ orientations} = 36 \text{ values}$.
4. **Total Feature Length:** $225 \times 36 = \mathbf{8,100}$ features.  
*Verified extracted shape:* `(400, 8100)`.

### HOG Feature Maps Across Blood Cell Classes:
![Q2 HOG Visualizations](assets/q2_hog_visualizations.png)

### LinearSVC Hyperparameter Tuning:
- $C = 0.1$: Val Acc = **70.83%**, Macro-F1 = **0.7078**
- **$C = 1.0$ (Selected Best):** **Val Acc = 71.67%**, **Macro-F1 = 0.7164**
- $C = 10.0$: Val Acc = **71.67%**, Macro-F1 = **0.7164**

### Validation Confusion Matrix:
![Q2 Confusion Matrix](assets/q2_hog_cm.png)

---

## Q3. SIFT + SVM Pipeline (7 marks)

### 1. Normal SIFT Keypoint Detection:
- Evaluated on all 400 training images (`cv2.SIFT_create().detectAndCompute`):
  - **Average Keypoints per Image:** **71.58**
  - **Minimum Keypoints:** **9** | **Maximum Keypoints:** **181** | **Zero-Keypoint Images:** **0**

![Q3 Normal SIFT Keypoints](assets/q3_sift_keypoints.png)

> **Why variable keypoints is a problem for SVM:**  
> Standard SVM classifiers require a fixed-length feature vector for every image instance. SIFT interest point detection generates variable-sized matrices ($k_i \times 128$) depending on local contrast, making them incompatible with linear hyperplanes without spatial pooling.

### 2. Dense Regular Grid SIFT & Hand Calculation:
- **Grid Size:** $4 \times 4 = 16$ keypoints (derived from last digit 2 $\in [0, 4]$)
- **Keypoint Coordinates:** Centers at $x, y \in \{16.0, 48.0, 80.0, 112.0\}$, keypoint size = 32
- **Descriptor Length:** 128 dimensions per keypoint
- **Total Feature Length:** $16 \text{ keypoints} \times 128 = \mathbf{2,048}$ features.  
*Verified extracted shape:* `(400, 2048)`.

![Q3 Dense Grid](assets/q3_dense_grid_keypoints.png)

### 3. LinearSVC Hyperparameter Tuning:
- **$C = 0.1$ (Selected Best):** **Val Acc = 82.50%**, **Macro-F1 = 0.8240**
- $C = 1.0$: Val Acc = **82.50%**, Macro-F1 = **0.8240**
- $C = 10.0$: Val Acc = **82.50%**, Macro-F1 = **0.8240**

### Validation Confusion Matrix:
![Q3 Confusion Matrix](assets/q3_sift_cm.png)

---

## Q4. Robustness: Rotation vs Noise (6 marks)

### Test Perturbations Evaluated:
![Q4 Transformations](assets/q4_transformations.png)

### Official Robustness Results Table (Test Set, 120 Images):
| Model Pipeline | Clean Accuracy | Rotated (+20°) Accuracy | Noisy ($\sigma=0.05$) Accuracy |
| :--- | :---: | :---: | :---: |
| **HOG + SVM** ($C=1.0$) | **65.00%** (0.6500) | **70.83%** (0.7083) | **57.50%** (0.5750) |
| **SIFT + SVM** ($C=0.1$) | **77.50%** (0.7750) | **76.67%** (0.7667) | **77.50%** (0.7750) |

### Robustness Comparison Bar Chart:
![Q4 Bar Chart](assets/q4_robustness_chart.png)

### Prediction Verification (Three Sentences):
1. Under Gaussian noise, our prediction was contradicted because **Dense SIFT proved remarkably resilient, maintaining 77.50% accuracy with zero degradation**, whereas HOG suffered a severe drop from 65.00% down to 57.50% (7.50% loss) due to additive noise corrupting boundary gradients.
2. Under rotation, **Dense SIFT remained highly stable with only a negligible drop** (77.50% to 76.67%), while HOG exhibited sample-dependent variance because rigid cell grids misalign with cell boundaries.
3. Overall, **Dense SIFT demonstrated superior baseline accuracy and environmental robustness across both perturbations**.

---

## Q5. Short Answers and AI Log (3 marks)

### Theoretical Short Answers:
- **(a) Why does HOG use block normalisation?**  
  *HOG uses block normalisation to evaluate gradient histograms across grouped, overlapping cells and scale them (e.g. via $L_2\text{-Hys}$) to make feature descriptors invariant to illumination changes, shadows, and staining contrast variations. Without block normalisation, variations in lighting or stain intensity directly alter gradient magnitudes, severely hurting classifier generalization across different lab samples.*
- **(b) Why is it wrong to choose C using the test set?**  
  *Choosing the regularization parameter $C$ on the test set causes data snooping and information leakage, as the model's configuration becomes implicitly fitted to the evaluation data. The test set must remain strictly unseen to serve as an unbiased benchmark of true generalization performance on new, independent patient samples.*

### Class Confusion Analysis:
- **Which two classes did your models confuse most, and why might grayscale make this harder?**  
  *The models confused **eosinophils (Class 1) and neutrophils (Class 6)** most frequently. In clinical cytology, both cell types share similar multilobed nuclei and granular cytoplasm. Their primary diagnostic distinction under standard Romanowsky staining is color: eosinophils contain distinctive bright reddish-orange granules, while neutrophils have neutral pink/lilac granules. Converting images to **grayscale strips away all color channels**, forcing the models to distinguish them purely by subtle shape and texture contours.*

### AI Log Table:
| Prompt (One Line) | What I Changed | How I Checked the Result Was Correct |
| :--- | :--- | :--- |
| *"How to place a 4x4 dense grid of SIFT keypoints with size 32 on 128x128 images?"* | Placed keypoint centers uniformly at `[16, 48, 80, 112]` along each axis and wrapped them as `cv2.KeyPoint(x, y, 32.0)`. | Printed the keypoint count (16) and descriptor feature length (2048 = 16 x 128), and visually plotted the grid overlay using `cv2.drawKeypoints`. |
| *"How to calculate HOG feature vector length by hand for 128x128 image with 8x8 cell size?"* | Applied block formula: `(16-2+1) * (16-2+1) * (2*2*9)` to verify block stepping across the grid. | Compared hand calculation (8,100) with `X_train_hog.shape[1]`, confirming an exact numerical match. |
| *"What is the proper way to add Gaussian noise with $\sigma=0.05$ to images scaled 0 to 1?"* | Normalized uint8 to float32 $[0, 1]$, sampled `np.random.normal(0, 0.05)`, clipped to $[0, 1]$, and converted back to uint8. | Verified pixel range bounds $[0, 255]$ and visualized clean vs noisy images side-by-side. |

---

## Viva Voce Preparation Guide (10 Key Questions & Answers)

1. **Q: What is the main difference between HOG and SIFT descriptors?**  
   **A:** HOG captures global shape by computing gradient orientation histograms across rigid, dense spatial grids of cells. SIFT captures local gradient structures around keypoints, utilizing scale and orientation invariance principles.

2. **Q: Why did your registration number (42) yield an 8x8 HOG cell size and a 4x4 SIFT grid?**  
   **A:** Registration number 42 ends in the digit 2. The exam rule states: even last digit gives $8 \times 8$ HOG cells, and digits 0 to 4 give a $4 \times 4$ SIFT grid (16 keypoints).

3. **Q: Walk me through the hand calculation for your HOG feature vector length of 8,100.**  
   **A:** A $128 \times 128$ image has $16 \times 16$ cells of size $8 \times 8$. With $2 \times 2$ cell blocks moving 1 cell at a time, there are $(16-2+1) \times (16-2+1) = 15 \times 15 = 225$ blocks. Each block has $2 \times 2 \times 9 = 36$ values. Thus, $225 \times 36 = 8,100$.

4. **Q: Why can't we directly feed standard SIFT keypoint descriptors into an SVM?**  
   **A:** Standard SIFT detects a variable number of keypoints per image depending on local contrast (in our data, from 9 to 181 keypoints). SVMs require fixed-dimension feature vectors for all samples.

5. **Q: How does a regular grid of keypoints solve this problem?**  
   **A:** By fixing the keypoint coordinates to a regular $4 \times 4$ grid (16 points), every image produces exactly 16 descriptors of 128 values each, resulting in a constant feature vector length of $16 \times 128 = 2,048$.

6. **Q: Which pipeline performed better overall, and why?**  
   **A:** SIFT + SVM performed significantly better (82.50% validation, 77.50% test) compared to HOG + SVM (71.67% validation, 65.00% test). SIFT's local sub-region gradient representation captures internal cytoplasmic texture and nuclear folds much better than HOG's rigid block histograms.

7. **Q: Why was SIFT so robust against 20-degree rotation and Gaussian noise?**  
   **A:** Dense SIFT extracts 128-dimensional normalized histograms over size-32 circular windows, which naturally averages high-frequency Gaussian noise. Keypoint descriptors remain stable across small rotations because gradients rotate smoothly within the local sub-regions.

8. **Q: Why did HOG accuracy drop by 7.5% under Gaussian noise?**  
   **A:** HOG computes direct 1-pixel horizontal and vertical finite-difference derivatives ($[-1, 0, 1]$). Pixel-level Gaussian noise strongly corrupts these 1st-order differences, altering orientation angles and dispersing counts across bins.

9. **Q: Why did the classifier confuse eosinophils and neutrophils most?**  
   **A:** Both are granulocytes with multilobed nuclei. In medicine, they are differentiated by granule color under stain (bright red/orange for eosinophils vs pale pink for neutrophils). Converting images to grayscale eliminates color channels, removing their primary distinguishing feature.

10. **Q: What is the role of hyperparameter C in LinearSVC?**  
    **A:** $C$ controls the trade-off between maximizing the classification margin and minimizing training classification errors. A smaller $C$ (like 0.1 for SIFT) enforces a wider, smoother margin that prevents overfitting on small sample sizes.
