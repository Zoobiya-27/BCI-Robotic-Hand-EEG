# eeg_utils.py
# Project: BCI-controlled Robotic Hand
# Date: 2025
# Description: Functions for loading, preprocessing, and extracting statistical features
#              from EEG motor imagery data using MNE and NumPy.

import numpy as np
import mne
from mne.datasets import eegbci
from typing import List, Dict

# Default channels and event IDs
DEFAULT_CHANNELS = ['C3', 'Cz', 'C4']
DEFAULT_EVENT_IDS = {'T1': 1, 'T2': 2}

def load_data(subject: int,
              runs: List[int],
              sfreq: int = 250,
              channels: List[str] = DEFAULT_CHANNELS,
              event_ids: Dict[str, int] = DEFAULT_EVENT_IDS,
              verbose: bool = True) -> mne.io.BaseRaw:
    """
    Load and preprocess EEG data from the PhysioNet EEG Motor Imagery dataset.

    Parameters
    ----------
    subject : int
        Subject number to load.
    runs : list of int
        List of run numbers.
    sfreq : int, optional
        Target sampling frequency for resampling (default 250 Hz).
    channels : list of str, optional
        EEG channels to pick (default ['C3', 'Cz', 'C4']).
    event_ids : dict, optional
        Dictionary mapping event names to IDs (default {'T1': 1, 'T2': 2}).
    verbose : bool, optional
        Print status messages if True.

    Returns
    -------
    raw : mne.io.BaseRaw
        Preprocessed raw EEG data.
    """
    if verbose:
        print(f"[INFO] Loading EEG data for subject {subject}...")

    # Load files from EEGBCI dataset
    files = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = mne.concatenate_raws(raws)

    # Standardize EEG channel names
    mne.datasets.eegbci.standardize(raw)

    # Pick specified channels
    raw.pick_channels(channels)

    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Resample to target frequency
    raw.resample(sfreq, npad='auto')

    if verbose:
        print(f"[INFO] Data loaded: {raw.n_channels} channels, {raw.n_times} samples.")

    return raw

def preprocess(raw: mne.io.BaseRaw,
               event_ids: Dict[str, int] = DEFAULT_EVENT_IDS,
               tmin: float = 0.0,
               tmax: float = 2.0,
               verbose: bool = True) -> mne.Epochs:
    """
    Preprocess EEG data: bandpass filter and epoching.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data.
    event_ids : dict, optional
        Event name to ID mapping (default {'T1':1, 'T2':2}).
    tmin : float, optional
        Start time before event (default 0.0s).
    tmax : float, optional
        End time after event (default 2.0s).
    verbose : bool, optional
        Print status messages if True.

    Returns
    -------
    epochs : mne.Epochs
        Epoched EEG data ready for feature extraction.
    """
    if verbose:
        print("[INFO] Filtering EEG data (8-30 Hz)...")

    # Bandpass filter to extract mu and beta rhythms
    raw.filter(8, 30, fir_design='firwin', verbose=False)

    if verbose:
        print("[INFO] Extracting events and epoching...")

    # Extract events from annotations
    events, _ = mne.events_from_annotations(raw, event_id=event_ids)

    # Create epochs
    epochs = mne.Epochs(raw, events, event_id=event_ids,
                        tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, verbose=False)

    if verbose:
        print(f"[INFO] Created {len(epochs)} epochs.")

    return epochs

def extract_stat_features(X_csp: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from CSP-filtered EEG data.

    Features per component:
        - Mean
        - Variance
        - Standard deviation
        - Maximum
        - Minimum
        - Median
        - Sum of absolute values

    Parameters
    ----------
    X_csp : np.ndarray
        CSP-filtered EEG data of shape (n_trials, n_components, n_samples).

    Returns
    -------
    features : np.ndarray
        Statistical feature matrix of shape (n_trials, n_components * 7).
    """
    # Compute statistical features
    mean = np.mean(X_csp, axis=2)
    var = np.var(X_csp, axis=2)
    std = np.std(X_csp, axis=2)
    max_ = np.max(X_csp, axis=2)
    min_ = np.min(X_csp, axis=2)
    median = np.median(X_csp, axis=2)
    sum_abs = np.sum(np.abs(X_csp), axis=2)

    # Concatenate features along component axis
    features = np.concatenate([mean, var, std, max_, min_, median, sum_abs], axis=1)

    return features
