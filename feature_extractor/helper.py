import logging
import os
import warnings
import antropy as ant
import numpy as np
import pandas as pd
import scipy.io
from mne_features.feature_extraction import extract_features
from scipy.stats import differential_entropy

# Logger for this module
logger = logging.getLogger(__name__)


def extract_eeg_features(data, selected_features, freq_bands, batch_size=1):
    """EEG feature extraction in batches."""
    freq_bands_list = list(freq_bands.values())

    funcs_params = None if any(f in selected_features for f in ["ptp_amp", "quantile", "rms"]) else {
        "pow_freq_bands__normalize": False,
        "pow_freq_bands__psd_method": "fft",
        "pow_freq_bands__freq_bands": freq_bands_list,
    }

    sfreq = 512
    features_list = []

    for i in range(0, len(selected_features), batch_size):
        batch = selected_features[i:i + batch_size]
        logger.info(f"Processing batch: {batch}")
        features_batch = extract_features(
            data, sfreq,
            selected_funcs=batch,
            return_as_df=True,
            funcs_params=funcs_params
        )
        features_list.append(features_batch)

    logger.info("Combining batch results...")
    return pd.concat(features_list, axis=1)


def load_eeg_data(filename):
    """Load EEG data from a .mat file."""
    data = scipy.io.loadmat(filename)
    return data["avedata"]


def extract_features_all_ch(eeg_data, selected_features, freq_bands, batch_size=4):
    reshaped_data = np.expand_dims(eeg_data, axis=0)
    extracted_features_df = extract_eeg_features(
        reshaped_data, selected_features, freq_bands, batch_size=batch_size
    )

    num_channels = eeg_data.shape[0]

    for i in range(num_channels):
        channel_data = eeg_data[i, :]
        de = differential_entropy(channel_data)
        ae = ant.app_entropy(channel_data, order=2, metric="chebyshev")
        extracted_features_df[f"channel_{i + 1}_differential_entropy"] = de
        extracted_features_df[f"channel_{i + 1}_approximate_entropy"] = ae

    return extracted_features_df


def save_features_per_file(features_df, subject_id, event_type, output_dir):
    subject_folder = os.path.join(output_dir, f"{subject_id}_{event_type}")
    os.makedirs(subject_folder, exist_ok=True)
    feature_file = os.path.join(subject_folder, f"{subject_id}_{event_type}_features.csv")
    features_df.to_csv(feature_file, index=False)
    logger.info(f"Saved features for {subject_id} ({event_type}) to {feature_file}")


def process_file(filename, data_dir, output_master_dir,
                 selected_features, batch_size, delete_channels, freq_bands):
    """Process a single EEG file."""
    if not filename.endswith(".mat"):
        logger.info(f"Skipping non-MAT file: {filename}")
        return None

    file_path = os.path.join(data_dir, filename)
    eeg_data = load_eeg_data(file_path)

    if delete_channels is None:
        warnings.warn("No channels specified to delete! Processing all channels.", UserWarning)
    elif isinstance(delete_channels, (list, tuple)):
        eeg_data = np.delete(eeg_data, delete_channels, axis=0)
        logger.info(f"Deleted channels: {delete_channels}")
    else:
        raise ValueError(
            f"Invalid type for delete_channels: {type(delete_channels)}. "
            "Expected list, tuple, or None."
        )

    features_df = extract_features_all_ch(eeg_data, selected_features, freq_bands, batch_size=batch_size)

    subject_id = filename.split(".")[0]
    event_type = "Extrovert" if filename.startswith("E") else "Introvert"
    logger.info(f"Processing subject: {subject_id} ({event_type})")

    features_df["SubjectID"] = subject_id
    features_df["EventType"] = event_type

    save_features_per_file(features_df, subject_id, event_type, output_master_dir)

    return features_df
