# train_csplda.py
"""
train_model.py
====================
Trains CSP + LDA models for BCI-controlled robotic hand using EEG motor imagery data.
- Loads EEG data for all training subjects
- Preprocesses EEG (filtering, epoching)
- Extracts CSP features
- Extracts statistical features for LDA
- Performs 5-fold cross-validation
- Saves CSP and LDA models for real-time robotic hand control
"""

import joblib
import numpy as np
from bci_utils import load_data, preprocess, extract_stat_features
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --- CONFIGURATION ---
ALL_SUBJECTS = list(range(1, 110))  # PhysioNet subjects 1-109
TEST_SUBJECT = 77                   # Subject reserved for testing/demo
TRAIN_SUBJECTS = [s for s in ALL_SUBJECTS if s != TEST_SUBJECT]
RUNS = [4, 8, 12]                   # EEG runs to use for training
CSP_COMPONENTS = 2                  # Number of CSP components

# --- TRAINING FUNCTION ---
def train_and_save():
    X_all, y_all = [], []

    print("🔹 Starting EEG data loading and preprocessing...")

    for subj in TRAIN_SUBJECTS:
        try:
            raw = load_data(subj, RUNS, sfreq=250)
            epochs = preprocess(raw)  # Bandpass, segmentation, artifact removal
            X = epochs.get_data()
            y = epochs.events[:, -1]  # Motor imagery labels
            X_all.append(X)
            y_all.append(y)
            print(f"✅ Loaded Subject {subj}")
        except Exception as e:
            print(f"❌ Failed Subject {subj}: {e}")

    # Combine all subjects' data
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    print(f"🔹 Total epochs loaded: {X_all.shape[0]}")

    # --- FEATURE EXTRACTION ---
    print("🔹 Extracting CSP features...")
    csp = CSP(n_components=CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X_all, y_all)
    X_feat = extract_stat_features(X_csp)  # Variance, mean, std, etc.

    # --- CLASSIFICATION ---
    print("🔹 Training LDA classifier with 5-fold CV...")
    lda = LinearDiscriminantAnalysis()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(lda, X_feat, y_all, cv=cv)
    print(f"🎯 Cross-validation accuracy: {np.mean(scores)*100:.2f}%")

    # Fit final LDA on all training data
    lda.fit(X_feat, y_all)

    # --- SAVE MODELS ---
    joblib.dump(csp, "csp_model.pkl")
    joblib.dump(lda, "lda_model.pkl")
    print("✅ Models saved: csp_model.pkl, lda_model.pkl")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    train_and_save()
