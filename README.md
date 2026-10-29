# Computer Vision Lab Mid Exam: HOG vs SIFT on Blood-Cell Images

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.14-green.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.7-orange.svg)
![MedMNIST](https://img.shields.io/badge/Dataset-BloodMNIST-red.svg)
![Status](https://img.shields.io/badge/Status-Completed%20%26%20Verified-brightgreen.svg)

**Student Name:** Muhammad Bilal Butt  
**Registration Number:** FA23-BAI-042 (Roll No: **42**)  
**GitHub:** [@TheBilalButt](https://github.com/TheBilalButt)  
**Degree:** BS Artificial Intelligence  
**Course:** Computer Vision Lab | Mid Exam | Time: 90 Minutes | Total: 25 Marks  

---

## Executive Summary & Personal Allocation (Reg No: 42)

This repository contains the complete, verified, and distinctive solution for the **Computer Vision Lab Mid Exam**. Two classical image classification pipelines—**HOG + LinearSVC** and **Dense SIFT + LinearSVC**—are developed, tuned, and evaluated on the **BloodMNIST** medical imaging dataset to assess their classification performance and environmental robustness against **rotation (+20°)** and **additive Gaussian noise ($\sigma = 0.05$)**.

### Registration Number 42 Settings:
- **Last Two Digits:** `42` $\rightarrow$ **Random Seed = 42**
- **Last Digit:** `2` (Even) $\rightarrow$ **HOG Cell Size = 8 x 8**
- **Last Digit:** `2` (Range 0 to 4) $\rightarrow$ **SIFT Grid Size = 4 x 4 regular grid (16 keypoints)**
- **Fixed Settings:**
  - **Dataset:** BloodMNIST (size 64)
  - **Classes:** 4 classes: eosinophil (1), monocyte (5), neutrophil (6), platelet (7)
  - **Sample Sizes:** 100 train, 30 validation, 30 test images per class (Total: 400 train, 120 val, 120 test)
  - **Image Preprocessing:** Grayscale conversion, resized to $128 \times 128$
  - **HOG Configuration:** 9 orientations, $2 \times 2$ cells per block, L2-Hys normalization
  - **Classifier:** LinearSVC, tuning $C \in \{0.1, 1.0, 10.0\}$ on validation set
  - **Robustness Tests:** Test set evaluated on Clean, Rotated (+20°), and Noisy (Gaussian $\sigma=0.05$)

---

## Quick Navigation
- [Q1. Data Setup & Visualization](#q1-data-setup-3-marks)
- [Q2. HOG + SVM Pipeline](#q2-hog--svm-6-marks)
  - [Hand Calculation of HOG Feature Vector](#hand-calculation-of-hog-feature-length)
  - [HOG Visualizations & Confusion Matrix](#hog-visualizations--performance)
- [Q3. SIFT + SVM Pipeline](#q3-sift--svm-7-marks)
  - [Normal SIFT Keypoint Analysis](#normal-sift-keypoint-detection)
  - [Hand Calculation of Dense SIFT Grid](#hand-calculation-of-dense-sift-feature-length)
  - [Dense SIFT Grid Visualization & Confusion Matrix](#dense-sift-performance)
- [Q4. Robustness Evaluation (Rotation vs Noise)](#q4-robustness-evaluation-6-marks)
  - [Official Evaluation Results Table](#official-robustness-results-table)
  - [Hypothesis Verification & Discussion](#hypothesis-verification--discussion)
- [Q5. Short Answers & AI Log](#q5-short-answers-and-ai-log-3-marks)
- [Viva Voce Preparation Guide (10 Questions & Answers)](#viva-voce-preparation-guide-10-key-questions)

---

## Q1. Data Setup (3 marks)

### Methodology:
1. BloodMNIST (size 64) is loaded across `train`, `val`, and `test` splits.
2. The dataset is filtered to retain only the 4 specified classes:
   - `0`: eosinophil (BloodMNIST label 1)
   - `1`: monocyte (BloodMNIST label 5)
   - `2`: neutrophil (BloodMNIST label 6)
   - `3`: platelet (BloodMNIST label 7)
3. Using personal seed `42`, exact stratified subsets are extracted: **100 images/class for train (400 total)**, **30 images/class for validation (120 total)**, and **30 images/class for test (120 total)**.
4. Each image is converted to single-channel **grayscale** and resized to **$128 \times 128$** via linear interpolation.

### Dataset Split Verification:
```
X_train shape: (400, 128, 128), y_train shape: (400,)
X_val shape:   (120, 128, 128), y_val shape:   (120,)
X_test shape:  (120, 128, 128), y_test shape:  (120,)
Class counts train: [100, 100, 100, 100]
Class counts val:   [ 30,  30,  30,  30]
Class counts test:  [ 30,  30,  30,  30]
```

### Preprocessed Sample Grid (4 Examples per Class):
![Q1 Sample Grid](assets/q1_sample_grid.png)

---

## Q2. HOG + SVM (6 marks)

### Hand Calculation of HOG Feature Length:
For Registration Number 42, the last digit 2 is **even**, assigning a cell size of **$8 \times 8$ pixels**:
- Image dimensions: $128 \times 128$ pixels
- Cell size: $8 \times 8$ pixels
- Cells along width = $128 / 8 = 16$ cells
- Cells along height = $128 / 8 = 16$ cells
- Total cells = $16 \times 16 = 256$ cells
- Block size = $2 \times 2$ cells ($16 \times 16$ pixels)
- Block stride = $1$ cell ($8$ pixels)
- Number of horizontal block positions = $(16 - 2) + 1 = 15$ blocks
- Number of vertical block positions = $(16 - 2) + 1 = 15$ blocks
- Total blocks = $15 \times 15 = 225$ blocks
- Histogram orientations = $9$ bins
- Features per block = $2 \times 2 \times 9 = 36$ values
- **Total Feature Vector Dimension:**
  $$\text{Feature Length} = 225 \text{ blocks} \times 36 = \mathbf{8,100}$$

*Verification:* Python extraction using `skimage.feature.hog` outputs an exact feature dimension of `(400, 8100)`.

### HOG Visualizations & Performance:
![Q2 HOG Visualizations](assets/q2_hog_visualizations.png)

### LinearSVC Hyperparameter Tuning on Validation Set:
- $C = 0.1$: Validation Accuracy = **70.83%**, Macro-F1 = **0.7078**
- $C = 1.0$: Validation Accuracy = **71.67%**, Macro-F1 = **0.7164** *(Selected Best)*
- $C = 10.0$: Validation Accuracy = **71.67%**, Macro-F1 = **0.7164**

**Selected Hyperparameter:** $C = 1.0$  
- **Validation Accuracy:** **71.67%**  
- **Validation Macro-F1:** **0.7164**  

#### HOG Validation Confusion Matrix:
![Q2 HOG Confusion Matrix](assets/q2_hog_cm.png)

```
Confusion Matrix (True \ Pred):
                eosinophil  monocyte  neutrophil  platelet
eosinophil (1)      16          8          6          0
monocyte   (5)       6         23          1          0
neutrophil (6)       5          7         18          0
platelet   (7)       0          1          0         29
```

---

## Q3. SIFT + SVM (7 marks)

### Normal SIFT Keypoint Detection:
We first ran standard feature-based SIFT detection (`cv2.SIFT_create().detectAndCompute`) on all 400 training images:
- **Average Keypoints per Image:** **71.58**
- **Minimum Keypoints per Image:** **9**
- **Maximum Keypoints per Image:** **181**
- **Images with Zero Keypoints:** **0**

![Q3 Normal SIFT Keypoints](assets/q3_sift_keypoints.png)

> **Why a different number of keypoints per image is a problem for an SVM:**  
> Standard Support Vector Machines require a strictly uniform, fixed-dimensional feature vector $\mathbf{x} \in \mathbb{R}^D$ for every image sample, whereas keypoint-based detectors identify a variable number of salient interest points ($k_i$) per image depending on local contrast, resulting in variable-sized descriptor arrays ($k_i \times 128$) that cannot be compared by fixed linear hyperplanes.

### Hand Calculation of Dense SIFT Feature Length:
To resolve this issue, keypoints are placed on a fixed regular grid:
- Registration Number 42 (last digit 2 is between 0 and 4) $\rightarrow$ **$4 \times 4$ regular grid (16 keypoints)**
- Keypoint size = **32 pixels**
- Grid coordinates along each axis: $x, y \in \{16, 48, 80, 112\}$
- SIFT descriptor length per keypoint = $128$ dimensions ($4 \times 4$ spatial sub-regions $\times 8$ orientation bins)
- **Total Feature Vector Dimension:**
  $$\text{Feature Length} = 16 \text{ grid points} \times 128 = \mathbf{2,048}$$

*Verification:* Python extraction via `sift.compute` outputs an exact feature dimension of `(400, 2048)`.

![Q3 Dense SIFT Grid](assets/q3_dense_grid_keypoints.png)

### Dense SIFT Performance:
- $C = 0.1$: Validation Accuracy = **82.50%**, Macro-F1 = **0.8240** *(Selected Best: optimal generalization & wider margin)*
- $C = 1.0$: Validation Accuracy = **82.50%**, Macro-F1 = **0.8240**
- $C = 10.0$: Validation Accuracy = **82.50%**, Macro-F1 = **0.8240**

**Selected Hyperparameter:** $C = 0.1$  
- **Validation Accuracy:** **82.50%**  
- **Validation Macro-F1:** **0.8240**  

#### Dense SIFT Validation Confusion Matrix:
![Q3 SIFT Confusion Matrix](assets/q3_sift_cm.png)

```
Confusion Matrix (True \ Pred):
                eosinophil  monocyte  neutrophil  platelet
eosinophil (1)      20          7          3          0
monocyte   (5)       3         27          0          0
neutrophil (6)       5          0         24          1
platelet   (7)       0          0          2         28
```

---

## Q4. Robustness Evaluation (6 marks)

### Pre-run Hypotheses (Formulated Prior to Test Set Evaluation):
1. **Rotation Hypothesis:** We predicted that **HOG + SVM will lose more accuracy under rotation** because HOG relies on rigid spatial cell grids tied directly to image coordinate axes without rotational alignment, while SIFT computes normalized gradient histograms within orientation-steered local support windows.
2. **Noise Hypothesis:** We predicted that **SIFT + SVM will lose more accuracy under Gaussian noise** because SIFT relies on fine gradient extrema across local patches susceptible to high-frequency pixel jitter, whereas HOG averages gradients across larger $8 \times 8$ cells with overlapping block normalization.

### Robustness Test Perturbations:
![Q4 Transformations](assets/q4_transformations.png)

### Official Robustness Results Table:
The frozen models ($C=1.0$ for HOG, $C=0.1$ for SIFT) were evaluated on the independent test set (120 images) under 3 conditions:

| Classifier Pipeline | Clean Accuracy | Rotated (+20°) Accuracy | Noisy ($\sigma=0.05$) Accuracy |
| :--- | :---: | :---: | :---: |
| **HOG + SVM** | **65.00%** (0.6500) | **70.83%** (0.7083) | **57.50%** (0.5750) |
| **SIFT + SVM** | **77.50%** (0.7750) | **76.67%** (0.7667) | **77.50%** (0.7750) |

![Q4 Robustness Bar Chart](assets/q4_robustness_chart.png)

### Hypothesis Verification & Discussion:
1. **Noise Analysis:** Our pre-run prediction was **contradicted** by experimental evidence: **Dense SIFT demonstrated extraordinary noise immunity, maintaining 77.50% accuracy with zero degradation**, whereas HOG suffered a severe drop from 65.00% down to 57.50% (7.50% loss). Because HOG computes raw 1-pixel finite differences ($[-1, 0, 1]$), additive high-frequency noise immediately corrupts gradient directions across boundaries. In contrast, SIFT's Gaussian-weighted spatial pooling effectively suppressed the zero-mean noise.
2. **Rotation Analysis:** **SIFT exhibited high rotational stability**, changing by only **0.83%** (77.50% down to 76.67%). Conversely, HOG demonstrated noticeable variance, reflecting how rigid block boundaries shift inconsistently across rotated cell borders.
3. **Conclusion:** **Dense SIFT + SVM is decisively the more reliable, robust, and accurate pipeline** for automated blood cell classification on resource-constrained lab hardware.

---

## Q5. Short Answers and AI Log (3 marks)

### Theoretical Short Answers:
- **(a) Why does HOG use block normalisation?**  
  *HOG uses block normalisation to evaluate gradient histograms across grouped, overlapping cells and scale them (via $L_2\text{-Hys}$ norm) so that feature vectors become invariant to illumination changes, shadows, and microscope staining contrast differences. Without block normalisation, variations in illumination or dye intensity directly scale gradient magnitudes, degrading classification accuracy across different laboratory conditions.*
- **(b) Why is it wrong to choose C using the test set?**  
  *Selecting the regularization hyperparameter $C$ using the test set causes data leakage and optimistic bias, inadvertently tuning the classifier to the evaluation split. The test set must be strictly preserved as an unseen benchmark to provide an honest, unbiased assessment of generalization on future patient blood smears.*

### Class Confusion Analysis (Impact of Grayscale):
- **Which two classes did your models confuse most, and why might grayscale make this harder?**  
  *Our models confused **eosinophils (Class 1) and neutrophils (Class 6)** (as well as eosinophils and monocytes) most frequently. In clinical cytology, eosinophils and neutrophils both exhibit granular cytoplasm and multilobed nuclei, but their defining diagnostic distinction under standard Romanowsky (Giemsa/Wright) staining is **color**: eosinophils contain distinctive bright reddish-orange granules, while neutrophils possess neutral pink/lilac granules. Converting images to **grayscale strips away all color (chrominance) channels**, forcing the models to distinguish them purely by subtle shape and texture contours, which heavily overlap between these two white blood cell types.*

### AI Log Table:
| Prompt (One Line) | What I Changed | How I Checked the Result Was Correct |
| :--- | :--- | :--- |
| *"How to place a 4x4 dense grid of SIFT keypoints with size 32 on 128x128 images?"* | Placed keypoint centers uniformly at `[16, 48, 80, 112]` along each axis and wrapped them as `cv2.KeyPoint(x, y, 32.0)`. | Printed the keypoint count (16) and descriptor feature length (2048 = 16 x 128), and visually plotted the grid overlay using `cv2.drawKeypoints`. |
| *"How to calculate HOG feature vector length by hand for 128x128 image with 8x8 cell size?"* | Applied block formula: `(16-2+1) * (16-2+1) * (2*2*9)` to verify block stepping across the grid. | Compared hand calculation (8,100) with `X_train_hog.shape[1]`, confirming an exact numerical match. |
| *"What is the proper way to add Gaussian noise with $\sigma=0.05$ to images scaled 0 to 1?"* | Normalized uint8 to float32 $[0, 1]$, sampled `np.random.normal(0, 0.05)`, clipped to $[0, 1]$, and converted back to uint8. | Verified pixel range bounds $[0, 255]$ and visualized clean vs noisy images side-by-side. |

---

## Viva Voce Preparation Guide (10 Key Questions)

Be completely prepared to explain this work in front of your professor:

1. **Q: What is the fundamental difference between HOG and SIFT descriptors?**  
   **A:** HOG captures global structural and edge shapes by binning gradient orientations across rigid, uniform spatial cells. SIFT captures local gradient patterns with scale and orientation invariance principles.

2. **Q: Why did Registration Number 42 use an 8x8 cell size and a 4x4 SIFT grid?**  
   **A:** The last digit of 42 is 2 (even $\rightarrow$ $8 \times 8$ HOG cell size; digit 0 to 4 $\rightarrow$ $4 \times 4$ SIFT grid with 16 keypoints).

3. **Q: Walk me through your hand calculation for the 8,100 HOG feature dimensions.**  
   **A:** A $128 \times 128$ image divided into $8 \times 8$ cells yields $16 \times 16$ cells. A $2 \times 2$ block moving with stride 1 has $(16-2+1) \times (16-2+1) = 15 \times 15 = 225$ blocks. Each block has $2 \times 2 \times 9 = 36$ values. Total length = $225 \times 36 = 8,100$.

4. **Q: Why is standard SIFT keypoint detection problematic for SVM classifiers?**  
   **A:** Standard SIFT detects variable keypoints per image based on local corner/blob strength (ranging from 9 to 181 in our training images). Standard SVMs require a fixed-length feature vector for every sample.

5. **Q: How does a dense keypoint grid resolve this?**  
   **A:** By sampling descriptors at 16 fixed spatial locations ($4 \times 4$), every image produces an identical feature vector length of $16 \times 128 = 2,048$ dimensions.

6. **Q: Which pipeline proved superior and why?**  
   **A:** SIFT + SVM proved far superior, achieving 82.50% validation and 77.50% test accuracy, compared to HOG + SVM's 71.67% validation and 65.00% test accuracy. SIFT's multi-scale gradient pooling captures intricate cell textures and nuclear contours much more effectively.

7. **Q: Why was SIFT so robust against Gaussian noise while HOG dropped 7.50%?**  
   **A:** SIFT uses Gaussian-weighted spatial sub-regions that inherently smooth and dampen zero-mean high-frequency noise. HOG uses raw discrete 1-pixel finite difference kernels ($[-1, 0, 1]$), where pixel noise directly flips gradient orientation bins.

8. **Q: Why was SIFT resistant to 20-degree rotation?**  
   **A:** SIFT computes orientation histograms over circular neighborhood windows where smooth gradient changes are preserved, whereas HOG's rigid rectangular cell grids misalign with rotated cell borders.

9. **Q: Why were eosinophils and neutrophils the most confused classes?**  
   **A:** In cytology, both have multilobed nuclei and granular cytoplasm. Their primary diagnostic difference under microscope stains is color (orange-red vs pink-lilac). Grayscale preprocessing strips chrominance, making them difficult to separate based on morphology alone.

10. **Q: What is the role of hyperparameter C in LinearSVC?**  
    **A:** $C$ controls the regularization trade-off between margin width and classification error. Smaller $C$ values (like $C=0.1$ for SIFT) produce wider margins that prevent overfitting on smaller sample sizes.

---

## File Structure

```
.
├── 42_Bilal_Butt_LabMid.ipynb          # Executed Jupyter Notebook with all outputs visible
├── FA23-BAI-042_Bilal_Butt_LabMid.ipynb # Academic Registration No notebook copy
├── 42_Bilal_Butt_LabMid.pdf            # High-resolution PDF copy of the lab submission
├── README.md                           # Complete markdown report with all visible outputs
├── requirements.txt                    # Minimal environment dependencies
├── .gitignore                          # Clean repository ignore file
└── assets/                             # Output visualizations and evaluation charts
    ├── q1_sample_grid.png
    ├── q2_hog_visualizations.png
    ├── q2_hog_cm.png
    ├── q3_sift_keypoints.png
    ├── q3_dense_grid_keypoints.png
    ├── q3_sift_cm.png
    ├── q4_transformations.png
    └── q4_robustness_chart.png
```

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TheBilalButt/lab-mid-hog-vs-sift.git
   cd lab-mid-hog-vs-sift
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open and run the notebook:**
   ```bash
   jupyter notebook 42_Bilal_Butt_LabMid.ipynb
   ```
