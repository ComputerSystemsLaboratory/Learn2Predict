#! /usr/bin/env python3

import os  
import yaml as yl  
import numpy as np  
import pandas as pd  

from joblib import dump  

from absl import app, flags, logging  

from sklearn.preprocessing import StandardScaler  


# -------------------------
# Define command-line flags
# ------------------------- 
FLAGS = flags.FLAGS  

flags.DEFINE_list(
    'directories', 
    None, 
    'List of paths to directories containing YAML files.'
)  

flags.DEFINE_string(
    'output_dir', 
    None, 
    'Output directory to save CSVs, structured by directories.'
)  

flags.DEFINE_list(
    'do_not_normalize', 
    [], 
    'List of keys to skip normalization (default: empty).'
)  

# Mark the 'directories' flag as required  
flags.mark_flag_as_required('directories')  


def load_all_data_by_directory(
    directories
):  
    """  
    Load all YAML data grouped by input directory and benchmark.  

    Args:  
        directories (list[str]): List of input directories to process.  

    Returns:  
        dict: A dictionary with keys as `directory/benchmark` pairs and values as DataFrames.  
        pd.DataFrame: A combined DataFrame for all data, used to fit the scaler.  
    """  
    all_data = []  # Global list for combined data (used for fitting the scaler)  
    grouped_data = {}  # Data organized by directory and benchmark  

    for directory in directories:  
        if not os.path.isdir(directory):  
            logging.error(f"The directory {directory} does not exist.")
            exit(1)
        
        for file_name in os.listdir(directory):  
            file_path = os.path.join(directory, file_name)  

            # Process only YAML files  
            if os.path.isfile(file_path) and file_name.endswith(('.yaml', '.yml')):  
                # Extract the benchmark name from the filename  
                try:  
                    benchmark_name = file_name.split('_', 1)[0]  # First part before '_'  
                except IndexError:  
                    logging.warning(f"Skipping file with invalid naming format: {file_name}")  
                    continue  

                # Load the YAML data  
                with open(file_path, 'r') as file:  
                    try:  
                        data = yl.safe_load(file) or {}  # Load dictionary (use empty dictionary if YAML is empty)  
                        if not isinstance(data, dict):  
                            logging.warning(f"Skipping non-dictionary file: {file_name}")  
                            continue  

                        # Add benchmark and directory metadata to the data  
                        data['__benchmark'] = benchmark_name  
                        data['__directory'] = os.path.basename(directory)  
                        data['__file'] = file_name  

                        # Organize by directory and benchmark  
                        group_key = (directory, benchmark_name)  
                        if group_key not in grouped_data:  
                            grouped_data[group_key] = []  
                        grouped_data[group_key].append(data)  

                        # Add to combined data for scaler fitting  
                        all_data.append(data)  
                    except yl.YAMLError as e:  
                        logging.error(f"Error reading YAML file {file_path}: {e}")  

    # Convert all combined data into a single DataFrame, ensuring missing keys are filled with 0.0  
    combined_df = pd.DataFrame(all_data).fillna(value=0.0)  

    # Convert grouped_data into DataFrame for each group, ensuring all keys are aligned and missing values are filled with 0.0  
    grouped_dataframes = {  
        key: pd.DataFrame(value).fillna(value=0.0) for key, value in grouped_data.items()  
    }  

    return grouped_dataframes, combined_df  


def process_yaml_files(  
    directories,  
    output_dir,  
    do_not_normalize  
):  
    """  
    Process YAML files from multiple directories, normalize the data (except specified columns), and create per-benchmark CSVs stored  
    in the corresponding subdirectories of the output directory. Save the scaler for each directory.  

    Args:  
        directories (list[str]): List of input directories to process.  
        output_dir (str): Root output directory for storing CSV files.  
        do_not_normalize (list[str]): List of keys to skip normalization.  
    """  
    # Step 1: Load and group all data  
    grouped_dataframes, combined_data = load_all_data_by_directory(directories)  

    if combined_data.empty:  
        logging.warning("No valid data found in the provided directories.")  
        return  

    # Identify columns to standardize (exclude metadata and do_not_normalize columns)  
    columns_to_standardize = [  
        col for col in combined_data.columns  
        if col not in do_not_normalize and col not in ['__benchmark', '__directory', '__file']  
    ]  

    # Filter numeric columns (StandardScaler only works on numeric data)  
    numeric_columns = combined_data[columns_to_standardize].select_dtypes(include=np.number).columns.tolist()  

    # Step 2: Fit the scaler on the combined dataset  
    scaler = StandardScaler()  
    if numeric_columns:  
        scaler.fit(combined_data[numeric_columns]) 
        logging.info(f"Fitted scaler using {len(numeric_columns)} numeric columns.")  
    
    # Save the scaler in each output directory  
    for directory in directories:  
        relative_dir = os.path.basename(directory)  
        target_dir = os.path.join(output_dir, relative_dir)  
        os.makedirs(target_dir, exist_ok=True)  
        scaler_file_path = os.path.join(target_dir, "scaler.scl")  
        dump(scaler, scaler_file_path)  # Save the scaler  
        logging.info(f"Saved scaler for directory '{relative_dir}' at '{scaler_file_path}'.")  

    # Step 3: Process each group (directory + benchmark)  
    for (directory, benchmark), df in grouped_dataframes.items():  
        # Standardize numeric columns  
        if numeric_columns:  
            df[numeric_columns] = scaler.transform(df[numeric_columns])  

        # Reorder columns  
        # General columns alphabetically  
        specific_columns = ['tasklets', 'dpus', 'pass', 'param']  
        existing_specific_columns = [col for col in specific_columns if col in df.columns]  # Retain only those that exist  
        other_columns = sorted([col for col in df.columns   
                                if col not in specific_columns   
                                and col not in do_not_normalize   
                                and col not in ['__benchmark', '__directory', '__file']])  
        
        # Add do_not_normalize columns at the end  
        normalize_excluded_columns = [col for col in do_not_normalize if col in df.columns]  

        column_order = other_columns + existing_specific_columns + normalize_excluded_columns  

        # Reorder the DataFrame columns  
        df = df[column_order]  

        # Determine the output path  
        relative_dir = os.path.basename(directory)  # Get just the name of the directory  
        target_dir = os.path.join(output_dir, relative_dir)  # Maintain directory structure in the output  
        os.makedirs(target_dir, exist_ok=True)  # Create the output directory if it doesn't exist  
        output_csv_path = os.path.join(target_dir, f"{benchmark}.csv")  

        # Save the CSV  
        df.to_csv(output_csv_path, index=False)  
        logging.info(f"Generated CSV at '{output_csv_path}' for benchmark '{benchmark}'.")  


# -------------------------------------------------
# Main routine
# -------------------------------------------------
def main(
    argv
):  
    """  
    Main entry point for the script. Processes YAML files, creates CSVs per benchmark, saves scalers,  
    and stores everything into structured directories under the output directory.  

    Args:  
        argv: Command-line arguments (not used directly in this function).  
    """  
    directories = FLAGS.directories  # List of input directories  
    output_dir = FLAGS.output_dir  # Root output directory for storing processed CSVs and scalers  
    do_not_normalize = FLAGS.do_not_normalize  # List of columns to skip normalization  

    # Create the output root directory if it doesn't exist  
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)  

    logging.info(f"Processing directories: {directories} with output_dir={output_dir}")  
    logging.info(f"Columns excluded from normalization: {do_not_normalize}")  

    process_yaml_files(
        directories, 
        output_dir, 
        do_not_normalize
    )  


if __name__ == '__main__':  
    app.run(main)  
