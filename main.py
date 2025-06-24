import argparse
import concurrent.futures
import logging
import os
import yaml
import pandas as pd
from tqdm import tqdm

from feature_extractor.combine_col_name import rename_column_levels
from feature_extractor.helper import process_file

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load config.yaml."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="EEG Feature Extraction")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")

    # Default multicore True unless --singlecore is provided
    parser.add_argument("--singlecore", action="store_true", help="Run in single-core mode (for debugging)")

    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker processes")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of features to process per batch")

    args = parser.parse_args()

    # Determine mode
    use_multicore = not args.singlecore

    if use_multicore:
        logger.info("Running in multi-core mode")
    else:
        logger.info("Running in single-core (debug) mode")

    config = load_config(args.config)

    freq_bands = config["freq_bands"]
    selected_features = config["selected_features"]
    data_dir = config["data_dir"]
    delete_channels = config.get("delete_channels", None)

    output_master_dir = os.path.join(data_dir, "extracted_features")
    os.makedirs(output_master_dir, exist_ok=True)

    all_features_list = []
    files = os.listdir(data_dir)
    mat_files = [f for f in files if f.endswith(".mat")]

    if use_multicore:
        logger.info("Starting multi-core processing")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    process_file, filename, data_dir, output_master_dir,
                    selected_features, args.batch_size, delete_channels, freq_bands
                ): filename
                for filename in mat_files
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
                result_df = future.result()
                if result_df is not None:
                    all_features_list.append(result_df)
    else:
        logger.info("Starting single-core processing")
        for filename in tqdm(mat_files, desc="Processing files"):
            result_df = process_file(
                filename, data_dir, output_master_dir,
                selected_features, args.batch_size, delete_channels, freq_bands
            )
            if result_df is not None:
                all_features_list.append(result_df)

    if all_features_list:
        df = pd.concat(all_features_list, ignore_index=True)

        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Renaming multi-index columns...")
            df.columns = [rename_column_levels(col) for col in df.columns]
            logger.info("Column renaming complete.")

        df.to_pickle(os.path.join(output_master_dir, "all_features_dataframe.pkl"))
        df.to_excel(os.path.join(output_master_dir, "all_features_dataframe.xlsx"), index=False)
    else:
        logger.warning("No features extracted or DataFrame could not be created.")

    logger.info("Feature extraction process completed.")


if __name__ == "__main__":
    main()
