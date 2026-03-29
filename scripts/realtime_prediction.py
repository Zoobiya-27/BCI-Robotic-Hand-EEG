# realtime_prediction.py
"""
Real-Time EEG Motor Imagery Classification + Robotic Hand Control
==================================================================
- Streams EEG data (simulated with PhysioNet or real-time)
- Applies CSP + LDA for classification
- Uses prediction smoothing (majority vote)
- Controls 3D-printed robotic hand via Raspberry Pi servos
"""

import numpy as np
import mne
from mne.datasets import eegbci
import joblib
import threading
import time
import queue
import RPi.GPIO as GPIO
import signal
import sys
from collections import deque

# ================= CONFIGURATION =================
TEST_SUBJECT = 77
RUNS = [4, 8, 12]
CHANNELS = ['C3', 'Cz', 'C4']
EVENT_IDS = {'T1': 1, 'T2': 2}        # T1: Left Hand, T2: Right Hand
SMOOTHING_WINDOW = 5                   # Majority vote over last N predictions
SERVO_PIN1 = 17
SERVO_PIN2 = 18
PWM_FREQ = 50

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN1, GPIO.OUT)
GPIO.setup(SERVO_PIN2, GPIO.OUT)

servo1 = GPIO.PWM(SERVO_PIN1, PWM_FREQ)
servo2 = GPIO.PWM(SERVO_PIN2, PWM_FREQ)

servo1.start(2.5)  # start position (open hand)
servo2.start(2.5)

def move_servos(angle):
    """Move servos to desired angle (0–180)."""
    duty = 2.5 + (angle / 18)
    servo1.ChangeDutyCycle(duty)
    servo2.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo1.ChangeDutyCycle(0)
    servo2.ChangeDutyCycle(0)

# ================= LOAD MODELS =================
csp = joblib.load("csp_model.pkl")
lda = joblib.load("lda_model.pkl")

# ================= FEATURE EXTRACTION =================
def extract_stat_features(X_csp):
    """Compute statistical features from CSP-transformed trials."""
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

# ================= THREADING QUEUES =================
raw_data_queue = queue.Queue()
prediction_queue = queue.Queue()

# ================= DATA LOADING & PREPROCESS =================
def load_data(subject, runs, sfreq=250):
    """Load EEG data from PhysioNet and preprocess channels."""
    files = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    raw.pick_channels(CHANNELS)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    raw.resample(sfreq, npad='auto')
    return raw

def preprocess(raw):
    """Bandpass filter and epoch EEG data."""
    raw.filter(8, 30, fir_design='firwin', verbose=False)
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_IDS)
    epochs = mne.Epochs(raw, events, event_id=EVENT_IDS, tmin=0, tmax=2,
                        baseline=None, preload=True, verbose=False)
    return epochs

# ================= THREAD FUNCTIONS =================
def data_acquisition(X, y):
    """Simulate real-time EEG streaming by putting each trial into the queue."""
    print("👾 Starting simulated real-time EEG streaming...")
    for i in range(len(X)):
        raw_data_queue.put((X[i], y[i]))
        time.sleep(1.0)  # adjust speed for demo
    raw_data_queue.put(None)  # signal end of streaming

def classification_thread():
    """Classify incoming EEG trials using CSP + LDA."""
    while True:
        item = raw_data_queue.get()
        if item is None:
            prediction_queue.put(None)
            break
        trial, true_label = item
        trial = trial[np.newaxis, :]               # shape (1, channels, samples)
        trial_csp = csp.transform(trial)
        features = extract_stat_features(trial_csp)
        prediction = lda.predict(features)[0]
        prediction_queue.put((prediction, true_label))

def output_thread():
    """Smooth predictions, print accuracy, and control robotic hand."""
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    correct, total = 0, 0
    last_action = None

    while True:
        result = prediction_queue.get()
        if result is None:
            acc = 100 * correct / total if total > 0 else 0
            print(f"\n🎯 Final Accuracy: {acc:.2f}%")
            cleanup()
            break

        pred, true = result
        prediction_history.append(pred)

        # Majority vote smoothing
        if len(prediction_history) == SMOOTHING_WINDOW:
            counts = {1:0, 2:0}
            for p in prediction_history:
                counts[p] += 1
            smoothed_pred = max(counts, key=counts.get)
        else:
            smoothed_pred = pred  # fallback if window not full

        total += 1
        if smoothed_pred == true:
            correct += 1

        action = "Open Hand" if smoothed_pred == 2 else "Close Hand"
        print(f"[{total}] Smoothed Prediction: {action} | Ground Truth: {true} | {'✅' if smoothed_pred == true else '❌'}")

        # Move servo only if action changed
        if action != last_action:
            move_servos(0 if smoothed_pred == 2 else 180)
            last_action = action

# ================= CLEANUP =================
def cleanup():
    """Stop PWM and cleanup GPIO."""
    print("\nCleaning up GPIO and exiting...")
    servo1.stop()
    servo2.stop()
    GPIO.cleanup()
    sys.exit(0)

def signal_handler(sig, frame):
    cleanup()

signal.signal(signal.SIGINT, signal_handler)

# ================= MAIN =================
if __name__ == "__main__":
    raw = load_data(TEST_SUBJECT, RUNS)
    epochs = preprocess(raw)
    X, y = epochs.get_data(), epochs.events[:, -1]

    # Start threads
    t1 = threading.Thread(target=data_acquisition, args=(X, y))
    t2 = threading.Thread(target=classification_thread)
    t3 = threading.Thread(target=output_thread)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()
