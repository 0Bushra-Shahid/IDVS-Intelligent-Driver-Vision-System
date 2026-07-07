# 🚗 IDVS — Intelligent Driver Vision System
> A real-time AI-powered driver monitoring system that uses a single webcam to simultaneously detect driver emotion, drowsiness, and distraction — and responds intelligently to each.

## 🎯 Problem Statement

Road accidents are one of the leading causes of death worldwide. Three major contributing factors are:
- **Driver drowsiness** — microsleep behind the wheel
- **Emotional fatigue** — stress, sadness affecting focus
- **Distraction** — looking away from the road
**IDVS addresses all three simultaneously using a single camera and AI.**

## 🧠 System Overview

IDVS consists of **3 independently trained modules** that are integrated into one unified real-time system:
Camera Input
     │
     ├──► FER Module      → Detect emotion → Play matching song
     ├──► MRL Module      → Detect drowsiness → Trigger alarm
     └──► Distraction     → Detect head pose → Trigger alert

## 📦 Modules

### 🎵 Module 1 — Facial Emotion Recognition (FER)

| Property | Details |
|----------|---------|
| Type | Supervised Learning |
| Task | Multi-class Classification |
| Algorithm | CNN (Convolutional Neural Network) |
| Classes | Happy, Neutral, Sad |
| Dataset | FER Dataset |
| Train Images | 31,213 |
| Test Images | 4,480 |
| Image Size | 48×48 Grayscale |
| Best Val Accuracy | **85%** |
| Output | Plays mood-matching music automatically |

**Class Performance:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Happiness | 0.93 | 0.92 | 0.92 |
| Neutral | 0.79 | 0.83 | 0.81 |
| Sadness | 0.82 | 0.77 | 0.80 |
| **Overall** | **0.85** | **0.85** | **0.85** |

---

### 👁️ Module 2 — Drowsiness Detection (MRL)

| Property | Details |
|----------|---------|
| Type | Supervised Learning |
| Task | Binary Classification |
| Algorithm | CNN (Convolutional Neural Network) |
| Classes | Open Eyes, Closed Eyes |
| Dataset | MRL Eye Dataset |
| Train Images | 8,104 |
| Test Images | 2,026 |
| Image Size | 64×64 Grayscale |
| Best Val Accuracy | **99.61%** |
| Output | Triggers alarm if eyes closed for 2+ seconds |

**Class Performance:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Open Eyes | 0.99 | 1.00 | 1.00 |
| Closed Eyes | 1.00 | 0.99 | 1.00 |
| **Overall** | **1.00** | **1.00** | **1.00** |

**Smart Features:**
- Normal blinks (< 1 second) are ignored
- Alarm only stops when eyes remain open for 1+ second continuously

### ⚠️ Module 3 — Distraction Detection

| Property | Details |
|----------|---------|
| Type | Rule-based Computer Vision |
| Algorithm | OpenCV Face Detection + Head Pose Analysis |
| Dataset | No dataset required |
| Output | Triggers alert if driver looks away for 2+ seconds |

**How it works:**
- Detects face using Haar Cascade
- Sets a baseline face size when driver looks forward
- If face size decreases significantly → driver looked away
- Tracks up/down movement via face position in frame

## 🔧 Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| TensorFlow 2.21 + Keras | Deep learning model building |
| OpenCV 4.13 | Real-time computer vision |
| Pygame | Audio playback (songs & alarms) |
| NumPy | Numerical operations |
| Pillow (PIL) | Image processing |
| Matplotlib + Seaborn | Data visualization |
| Scikit-learn | Model evaluation & metrics |


## 📁 Project Structure

IDVS/
├── FER Model/
│   ├── Data/
│   │   ├── raw/               ← Original FER dataset
│   │   └── processed/         ← Cleaned & preprocessed data
│   ├── model/
│   │   └── fer_best_model.keras
│   ├── songs/
│   │   ├── happiness/         ← Add happy mp3 files here
│   │   ├── neutral/           ← Add calm mp3 files here
│   │   └── sadness/           ← Add sad mp3 files here
│   ├── cleaning.ipynb
│   ├── statistics.ipynb
│   ├── preprocessing.ipynb
│   ├── model_train.ipynb
│   └── real_time_detection.ipynb
│
├── MRL Model/
│   ├── Data/
│   │   ├── raw/
│   │   │   ├── open_eyes/     ← Original MRL open eye images
│   │   │   └── closed_eyes/   ← Original MRL closed eye images
│   │   └── processed/
│   ├── model/
│   │   └── mrl_best_model.keras
│   ├── alarm/                 ← Add alarm mp3/wav file here
│   ├── cleaning.ipynb
│   ├── statistics.ipynb
│   ├── preprocessing.ipynb
│   ├── model_train.ipynb
│   └── real_time_detection.ipynb
│
├── Distraction Model/
│   ├── alarm/                 ← Add alarm mp3/wav file here
│   └── distraction_detection.ipynb
│
├── Integration/
│   └── integrated_system.ipynb   ← Run this for full system
│
└── README.md


## 📊 Complete ML Pipeline

Each module followed a complete ML pipeline from scratch:

Raw Dataset
     ↓
1. Cleaning
   - Remove corrupt images
   - Remove duplicate images (hash-based)
   - Remove blurry images (Laplacian variance)
     ↓
2. Statistics & EDA
   - Image size distribution
   - RGB vs Grayscale analysis
   - Pixel intensity histograms per class
   - Class balance (pie charts)
   - Mean, Median, Mode, Std Dev, Skewness
     ↓
3. Preprocessing
   - Convert to Grayscale
   - Resize (FER: 48x48, MRL: 64x64)
   - Augmentation (rotation, flip, zoom, shift)
   - Normalize pixel values (0-1)
   - Train/Test split (80/20)
     ↓
4. Model Training
   - CNN architecture with multiple conv blocks
   - BatchNormalization + Dropout (overfit prevention)
   - L2 Regularization
   - Label Smoothing (FER only)
   - Class Weights (imbalance handling)
   - EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
     ↓
5. Evaluation
   - Accuracy & Loss curves
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1)
     ↓
6. Real-time Detection
     ↓
7. Integration (all 3 modules together)

## 🏗️ CNN Architecture

### FER Model
- **3 Conv Blocks** (32 → 64 → 128 filters)
- Each block: Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout(0.25)
- Fully Connected: Dense(256) → BatchNorm → Dropout(0.5) → Dense(128) → BatchNorm → Dropout(0.5)
- Output: Dense(3, softmax)
- **Total Parameters: 1.5M**

### MRL Model
- **3 Conv Blocks** (32 → 64 → 128 filters)
- Same structure as FER
- Output: Dense(1, sigmoid) — binary classification
- **Total Parameters: 1.18M**

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/0Bushra-Shahid/IDVS-Intelligent-Driver-Vision-System.git
cd IDVS-Intelligent-Driver-Vision-System
```

### 2. Create virtual environment
```bash
python -m venv ml_venv
ml_venv\Scripts\activate       # Windows
source ml_venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install tensorflow keras opencv-python pygame numpy pillow matplotlib seaborn scikit-learn scipy
```

### 4. Add songs
```
FER Model/songs/happiness/  → Add upbeat mp3 files
FER Model/songs/neutral/    → Add calm mp3 files
FER Model/songs/sadness/    → Add slow mp3 files
```

### 5. Add alarm sounds
```
MRL Model/alarm/            → Add alarm mp3/wav file
Distraction Model/alarm/    → Add alarm mp3/wav file
```

### 6. Download dataset (optional — to retrain models)
> 📂 Dataset Download: [ https://drive.google.com/drive/folders/1nMS_z4oXOfIQ42pOLcKgVr4n9dYLM2lV?usp=sharing ] 

### 7. Run integrated system
```
Open Integration/integrated_system.ipynb
Run all cells
Press Q to quit camera window

## ✨ Key Features

- ✅ **Real-time performance on CPU** — no GPU required
- ✅ **Single webcam** — no additional hardware needed
- ✅ **Three independently trained modules** integrated into one system
- ✅ **Automatic song playback** based on detected emotion
- ✅ **Smart blink detection** — normal blinks are ignored, only prolonged eye closure triggers alarm
- ✅ **Alarm persistence** — alarm only stops when eyes are fully open for 1+ second
- ✅ **Distraction detection without any dataset** — pure computer vision
- ✅ **Overfit prevention** — Dropout, BatchNormalization, L2 Regularization, EarlyStopping
- ✅ **Class imbalance handled** — Augmentation + Class Weights


## 📈 Results Summary

| Module | Accuracy | Classes | Method |
|--------|----------|---------|--------|
| FER | 85% | 3 (Happy, Neutral, Sad) | CNN + Supervised Learning |
| MRL | 99.61% | 2 (Open, Closed) | CNN + Supervised Learning |
| Distraction | Rule-based | Forward / Away | OpenCV |

## 🚧 Limitations & Future Work

- FER sadness/neutral confusion can occur in ambiguous expressions
- Distraction detection works best with frontal face — side profiles may reduce accuracy
- Future: Add PERCLOS (Percentage of Eye Closure) for more robust drowsiness scoring
- Future: Add anger/fear emotion classes for more comprehensive FER
- Future: Deploy on Raspberry Pi for embedded vehicle use


## 👩‍💻 Author

**Bushra Shahid**
BS Computer Science Student
Passionate about Machine Learning and Computer Vision


## 📄 Disclaimer

This project is for **educational purposes only**. It is a student project and should not be used as a standalone safety-critical system in real vehicles without further validation and testing.


## 🙏 Acknowledgements

- FER Dataset — facial emotion recognition benchmark dataset
- MRL Eye Dataset — machine learning based eye state detection dataset
- TensorFlow & Keras — deep learning framework
- OpenCV — computer vision library
