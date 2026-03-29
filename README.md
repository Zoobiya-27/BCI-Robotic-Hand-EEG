# BCI-Controlled Robotic Hand for Motor-Impaired Individuals
### Final Year Project — Undergraduate

This repository contains the code, data, and models for our undergraduate final year project: a **Brain-Computer Interface (BCI)-controlled robotic hand** using EEG motor imagery signals. The system performs EEG acquisition, preprocessing, feature extraction (CSP), classification (LDA), and real-time robotic actuation. The project was implemented using **Python** for machine learning and real-time processing, and **MATLAB** for signal processing and simulation.

---

## Background

Motor imagery-based BCIs translate brain activity into commands without requiring physical movement, making them highly valuable in assistive robotics and rehabilitation. In this project, EEG signals corresponding to **left and right hand motor imagery** were recorded, classified in real time, and used to control a low-cost robotic hand via a Raspberry Pi.

The **CSP + LDA pipeline** was chosen for its low latency and robust performance in two-class motor imagery tasks, making it more suitable for real-time applications than computationally expensive deep learning models.

---

## Project Overview

### Phase 1 — Model Training with Public Dataset
We trained a binary classification model using the **PhysioNet Motor Imagery EEG Dataset**.

- Subject 85's data was used for training the real-time model
- Achieved **80% real-time accuracy** controlling a robotic hand via Raspberry Pi
- Achieved **62% overall accuracy** across the full dataset
- Controlled a **3D-printed robotic hand** using classification outputs

### Phase 2 — Data Collection & ERP Analysis
EEG data was collected using **gold cup electrodes** and the **OpenBCI Ganglion board**.

- Conducted left/right hand motor imagery tasks
- Stored raw EEG in CSV format via Raspberry Pi
- Performed **ERP analysis in MATLAB EEGLAB** to identify positive and negative deflections for left/right tasks

> This repository validates the ML model using the PhysioNet dataset due to noise in real-time acquisition, but the control system is designed for real-world operation.

---

## Methods

### Signal Acquisition
- **Phase 1:** Public EEG dataset (PhysioNet, 64 channels, motor imagery tasks)
- **Phase 2:**
  - EEG: OpenBCI Ganglion board (gold cup electrodes)
  - Sampling rate: 250 Hz
  - Electrodes placed at **C3, Cz, and C4**
  - Tasks designed and presented using **PsychoPy**

### Preprocessing (MATLAB)
- Bandpass filtering (8–30 Hz) to extract **mu and beta rhythms**
- Segmentation based on stimulus triggers
- Artifact removal (manual/automatic)
- ERP analysis in EEGLAB to detect motor-related potentials
- Exported segmented trials as `.npy` for Python classification

### Feature Extraction (Python)
- **Common Spatial Patterns (CSP)** for spatial filtering
- Variance of CSP-filtered signals as feature vectors

### Classification
- **Linear Discriminant Analysis (LDA)** for binary classification
- Real-time inference tested with Raspberry Pi controlling robotic hand

---

## Results

| Metric | Result |
|---|---|
| Real-time accuracy (Subject 85) | 80% |
| Overall dataset accuracy | 62% |
| ERP Analysis | Positive/negative peaks identifiable for L/R imagery |

The CSP+LDA pipeline provided **low-latency classification**, enabling smooth robotic control during demonstration.

---

## Limitations

- Real-time EEG acquisition produced noisy signals, limiting accuracy
- Final live demo used pre-trained model data
- Currently supports only **binary motor imagery classification**
- 3D-printed robotic hand was limited in grip strength

---

## Repository Structure

| File | Description |
|---|---|
| `eeg_utils.py` | Common helper functions for EEG loading, preprocessing, and feature extraction |
| `train_csplda.py` | Trains CSP + LDA models on all subjects; saves `csp_model.pkl` and `lda_model.pkl` |
| `demo_robotic_hand.py` | Loads pretrained models and runs real-time simulation on test subject data |
| `realtime_prediction.py` | Real-time prediction script |
| `csp_model.pkl` | Pretrained CSP model |
| `lda_model.pkl` | Pretrained LDA model |

---

## How to Run

**I. Preprocess EEG Data**
```bash
python eeg_utils.py
```

**II. Train Classification Model**
```bash
python train_csplda.py
```

**III. Run Real-time Prediction**
```bash
realtime_prediction.py
```

**IV. Demonstrate Robotic Control**
```bash
demo_robotic_hand.py
```

---

## Technologies Used

- **EEG Hardware:** OpenBCI Ganglion board
- **Signal Processing:** MATLAB, EEGLAB
- **Machine Learning:** Python, MNE, scikit-learn
- **Task Design:** PsychoPy
- **Robotic Control:** Raspberry Pi
- **Hardware:** 3D-printed robotic hand

---

## Impact

This system provides a **cost-effective assistive technology** that integrates real-time brain signal processing with robotic actuation — aimed at improving mobility and independence for motor-impaired individuals.

---

## Team

- **Zoobiya Aalam**
- Reehab Ahmed
- Aliza Shabraiz
