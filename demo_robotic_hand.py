# demo_robotic_hand.py
"""
Demo: Real-Time EEG Motor Imagery Prediction
=============================================
- Streams selected EEG trials in real-time
- Applies CSP + LDA classification
- Prints predictions with ground truth
- Can be extended to control robotic hand via GPIO
"""

import numpy as np
import mne
from mne.datasets import eegbci
import joblib
import threading
import time
import queue

# ================= CONFIGURATION =================
TEST_SUBJECT = 77
RUNS = [4, 8, 12]
CHANNELS = ['C3', 'Cz', 'C4']
EVENT_IDS = {'T1': 1, 'T2': 2}  # T1: Left hand, T2: Right hand

# ================= LOAD PRETRAINED MODELS =================
csp = joblib.load("csp_model.pkl")
lda = joblib.load("lda_model.pkl")

# ================= QUEUES =================
raw_data_queue = queue.Queue()
prediction_queue = queue.Queue()

# ================= DATA LOADING & PREPROCESS =================
def load_data(subject, runs, sfreq=250):
    """Load EEG data for demo."""
    files = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    raw.pick_channels(CHANNELS)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    raw.resample(sfreq, npad='auto')
    return raw

def preprocess(raw):
    """Filter and epoch EEG trials."""
    raw.filter(8, 30, fir_design='firwin', verbose=False)
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_IDS)
    epochs = mne.Epochs(raw, events, event_id=EVENT_IDS, tmin=0, tmax=2,
                        baseline=None, preload=True, verbose=False)
    return epochs

# ================= FEATURE EXTRACTION =================
def extract_stat_features(X_csp):
    """Extract simple statistical features from CSP-transformed data."""
    features = []
    for trial in X_csp:
        trial_feats = []
        for comp in trial:
            trial_feats.extend([
                np.mean(comp), np.var(comp), np.std(comp),
                np.max(comp), np.min(comp), np.median(comp),
                np.sum(np.abs(comp))
            ])
        features.append(trial_feats)
    return np.array(features)

# ================= THREAD FUNCTIONS =================
def data_acquisition(X, y, indices=[0, 1]):
    """Simulate streaming of selected EEG trials."""
    print("👾 Starting demo EEG streaming...")
    for i in indices:
        raw_data_queue.put((X[i], y[i]))
        time.sleep(1.5)  # simulate delay between trials
    raw_data_queue.put(None)

def classification_thread():
    """Classify streamed EEG trials using CSP + LDA."""
    while True:
        item = raw_data_queue.get()
        if item is None:
            prediction_queue.put(None)
            break
        trial, true_label = item
        trial = trial[np.newaxis, :]
        trial_csp = csp.transform(trial)
        features = extract_stat_features(trial_csp)
        prediction = lda.predict(features)[0]
        prediction_queue.put((prediction, true_label))

def output_thread():
    """Display predictions with ground truth and accuracy."""
    correct = 0
    total = 0
    while True:
        result = prediction_queue.get()
        if result is None:
            acc = 100 * correct / total if total > 0 else 0
            print(f"\n🎯 Demo Accuracy: {acc:.2f}%")
            break
        pred, true = result
        total += 1
        correct += int(pred == true)
        action = "Open Hand" if pred == 2 else "Close Hand"
        print(f"[{total}] Prediction: {action} | Ground Truth: {true} → {'✅' if pred == true else '❌'}")
        # 🔧 Add GPIO/servo commands here if needed

# ================= MAIN =================
if __name__ == "__main__":
    # Load and preprocess EEG data
    raw = load_data(TEST_SUBJECT, RUNS)
    epochs = preprocess(raw)
    X, y = epochs.get_data(), epochs.events[:, -1]

    # Select first two demo trials
    demo_indices = [41, 0]  # customize as needed

    # Launch threads
    t1 = threading.Thread(target=data_acquisition, args=(X, y, demo_indices))
    t2 = threading.Thread(target=classification_thread)
    t3 = threading.Thread(target=output_thread)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()
