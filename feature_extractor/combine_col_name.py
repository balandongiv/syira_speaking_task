import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

selected_features = [
    'line_length', 'kurtosis', 'ptp_amp', 'skewness', 'pow_freq_bands',
    'compute_mean', 'compute_variance', 'compute_std', 'compute_kurtosis',
    'compute_skewness', 'compute_hjorth_mobility', 'compute_hjorth_complexity',
    'compute_higuchi_fractal_dimension', 'compute_petrosian_fractal_dimension',
    'compute_katz_fractal_dimension', 'compute_energy', 'compute_teager_kaiser_energy',
    'compute_mean_abs_diff', 'compute_wavelet_coef_energy', 'compute_wavelet_coef_std',
    'compute_hurst_exponent', 'compute_svd_entropy', 'compute_svd_fisher_info',
    'compute_dfa', 'compute_median', 'compute_mean_abs', 'compute_zero_crossings',
    'compute_hjorth_parameters', 'compute_mean_frequency', 'compute_band_power',
    'compute_spectral_entropy', 'compute_spectral_edge_frequency', 'compute_spect_entropy',
    'compute_spect_kurtosis', 'compute_spect_skewness', 'compute_spect_slope',
    'compute_wavelet_energy', 'compute_wavelet_std', 'compute_wavelet_entropy',
    'compute_wavelet_coef', 'compute_wavelet_var', 'compute_corr',
    'compute_phase_locking_value', 'compute_phase_lag_index', 'compute_coherence',
    'compute_granger_causality'
]

# Define frequency band mappings
FREQ_BANDS = {
    "band0": "delta",
    "band1": "theta",
    "band2": "alpha",
    "band3": "sigma",
    "band4": "beta"
}

def rename_column_levels(col_tuple):
    """
    Renames a two-level column name according to specified rules:
    - Process the second-level name:
        * If it contains '/', convert it to '_div_' format.
        * Replace band identifiers (bandX) with the actual frequency names.
    - Returns the concatenated single-level column name.
    """
    first_level, second_level = col_tuple

    # Process second level name if present
    if second_level:
        # Handle division case: e.g., "ch12_band1/band2" -> "ch12_band1_div_band2"
        if "/" in second_level:
            parts = second_level.split("/")
            if len(parts) == 2:
                second_level = f"{parts[0]}_div_{parts[1]}"

        # Replace band identifiers with actual frequency names
        for band, freq in FREQ_BANDS.items():
            if band in second_level:
                second_level = second_level.replace(band, freq)

    # Combine first and second level properly
    return f"{first_level}_{second_level}" if second_level else first_level

def main():

    # Define paths
    main_path = r"D:\dataset\syira_speaking_task\extracted_features"
    file_path = os.path.join(main_path, "all_features_dataframe.pkl")
    excel_path = os.path.join(main_path, "all_features_dataframe.xlsx")
    pickle_path = os.path.join(main_path, "all_features_dataframe_renamed.pkl")  # Optional pickle output
    # Load the dataset with error handling
    try:
        logger.info(f"Loading dataset from: {file_path}")
        df = pd.read_pickle(file_path)
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Process only if columns are a MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        logger.info("Renaming multi-index columns...")
        df.columns = [rename_column_levels(col) for col in df.columns]
        logger.info("Column renaming complete.")
    else:
        logger.info("DataFrame columns are not MultiIndex, skipping renaming.")

    # Save to Excel with error handling
    try:
        df.to_excel(excel_path, index=False)
        logger.info(f"Data successfully saved to Excel: {excel_path}")
    except Exception as e:
        logger.error(f"Error saving to Excel: {e}")

    # Save to Pickle with error handling (optional)
    try:
        df.to_pickle(pickle_path)
        logger.info(f"Data successfully saved to Pickle: {pickle_path}")
    except Exception as e:
        logger.error(f"Error saving to Pickle: {e}")

if __name__ == "__main__":
    main()
