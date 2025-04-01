import os  
import re  
import numpy as np 
import pandas as pd  

from absl import logging  

from scipy.stats import shapiro

from sklearn.metrics import r2_score  
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 


# ---------------------------------------------
# Helper function: Log file to data
# ---------------------------------------------
def parse_log_file(
    filename, 
    baseline=0.0
):  
    """  
    Parses a log file and extracts information into a dictionary.  
    
    Args:  
        filename (str): Path to the file with the specific pattern.  
        baseline (float): A baseline value for calculating the speedup.  

    Returns:  
        dict: A dictionary with extracted keys ('tasklets', 'dpus', 'pass', 'param', 'time', 'speedup'),  
              or None if the file has an error (e.g., total_time <= 0).  
    """  
    # (1) Parse the filename  
    filename_pattern = r"UPMEM_.*?_tasklets_(\d+)_dpus_(\d+)_pass_(\d+)_param_(\d+)\.log"  
    match = re.search(filename_pattern, filename)  
    if not match:  
        logging.error("Filename does not match the required pattern.")  
        exit(1)

    # Extract values from the filename  
    tasklets = int(match.group(1))  
    dpus = int(match.group(2))  
    pass_num = int(match.group(3))  
    param = int(match.group(4))  
    
    # Initialize the dictionary with extracted values  
    result = {  
        "tasklets": tasklets,  
        "dpus": dpus,  
        "pass": pass_num,  
        "param": param,  
        "runtime": 0.0,  # Placeholder for DPU Kernel Time  
        "speedup": None  # Placeholder for speedup  
    }  

    # (2) Process the file to extract DPU Kernel Time  
    reduction_time = 0.0  
    scan_time = 0.0  
    add_time = 0.0  
    step_times = []  
    total_time = 0.0  

    with open(filename, 'r') as file:  
        lines = file.readlines()  

        # Scan all lines to search for patterns  
        for line in lines:  
            # Pattern 1: Reduction Time and Scan Time  
            match_reduction = re.search(r"DPU Kernel Reduction Time:\s*([\d.]+)\s*ms", line)  
            if match_reduction:  
                reduction_time += float(match_reduction.group(1))  
            
            match_scan = re.search(r"DPU Kernel Scan Time:\s*([\d.]+)\s*ms", line)  
            if match_scan:  
                scan_time += float(match_scan.group(1))  
            
            # Pattern 2: Scan Time and Add Time  
            match_add = re.search(r"DPU Kernel Add Time:\s*([\d.]+)\s*ms", line)  
            if match_add:  
                add_time += float(match_add.group(1))  
            
            # Pattern 3: Single Kernel Time (directly on a line)  
            match_kernel = re.search(r"DPU Kernel Time:\s*([\d.]+)\s*ms", line)  
            if match_kernel:  
                total_time += float(match_kernel.group(1))  
            
            # Pattern 4: Step-wise Kernel Times  
            step_kernel = re.search(r"Step \w+ DPU Kernel Time:\s*([\d.]+)\s*ms", line)  
            if step_kernel:  
                step_times.append(float(step_kernel.group(1)))  

    # Calculate the total kernel time from the matched components  
    # Sum the Reduction + Scan time (if found)  
    total_time += reduction_time + scan_time  
    # Add the Add time (if found)  
    total_time += add_time  
    # Add the Step-wise times (if found)  
    total_time += sum(step_times)  

    # If total_time <= 0, file has an error; return None  
    if total_time <= 0:
        return None  

    # Update the "time" key in the result dictionary  
    result["runtime"] = total_time  

    # Calculate speedup (baseline / time)  
    result["speedup"] = baseline / total_time  if baseline > 0.0 else total_time

    return result  


# ---------------------------------------------
# Helper function: Return specific files
# ---------------------------------------------     
def upmem_log_files(
    directory, 
    benchmark
):  
    """  
    Reads all files from a directory that match the pattern:  
    UPMEM_<benchmark>_tasklets_*_dpus_*_pass_*_param_*.log  
    
    Args:  
        directory (str): Path to the directory.  
        benchmark (str): The benchmark string to match in the pattern.  
    
    Returns:  
        list: A list of filenames that match the pattern.  
    """  
    # Ensure the directory exists  
    if not os.path.isdir(directory):  
        logging.error(f"The directory {directory} does not exist.")  
        exit(1)
    
    # Define the regex pattern for matching the filenames  
    pattern = rf"UPMEM_{benchmark}_tasklets_\d+_dpus_\d+_pass_\d+_param_\d+\.log"  
    
    # List to store matched filenames  
    matched_files = []  
    
    # Iterate through each file in the directory  
    for filename in os.listdir(directory):  
        # Match the filename against the pattern  
        if re.match(pattern, filename):  
            matched_files.append(filename)  
    
    return matched_files  


# ---------------------------------------------
# Helper function: Filter a subtring pattern
# ---------------------------------------------     
def extract_x86Histogram_embeddings_info(
    filename
):  
    """  
    Extracts the substring tasklets_{T}_dpus_{D}_pass_{P} from a given filename.  

    Args:  
        filename (str): The filename to extract the substring from.  

    Returns:  
        str: The matched substring, or None if no match is found.  
    """  
    pattern = r"tasklets_\d+_dpus_\d+_pass_\d+"  
    match = re.search(pattern, filename)  
    if match:  
        return match.group(0)  # Return the matched substring  
    return None  


# ---------------------------------------------
# Helper function: Filter a subtring pattern
# ---------------------------------------------     
def extract_upmem_log_info(
    filename
):  
    """  
    Extracts the substring tasklets_{T}_dpus_{D}_pass_{P} from a given filename.  

    Args:  
        filename (str): The filename to extract the substring from.  

    Returns:  
        str: The matched substring, or None if no match is found.  
    """  
    pattern = r"tasklets_\d+_dpus_\d+_pass_\d+_param_\d+"  
    match = re.search(pattern, filename)  
    if match:  
        return match.group(0)  # Return the matched substring  
    return None  


# ---------------------------------------------
# Helper function: Capitalize first letter
# --------------------------------------------- 
def capitalize_first_letter(
    input_string
):  
    """  
    Capitalizes the first letter of the input string.  

    Parameters:  
        input_string (str): The input string.  

    Returns:  
        str: The string with the first letter capitalized.  
    """  
    if not input_string:  
        return "Input string is empty."  
    return input_string[0].upper() + input_string[1:]  


# ---------------------------------------------
# Helper function: Save the residuals
# --------------------------------------------- 
def save_residuals_to_file(
    y_test, 
    y_pred, 
    output_dir: str = ".", 
    output_filename: str = "residuals_data.csv"
):  
    """  
    Save y_test, y_pred, and residuals to a CSV file.  

    Parameters:  
        y_test (np.ndarray): Array of true target (ground truth) values.  
        y_pred (np.ndarray): Array of predicted values from the model.  
        output_dir (str): Directory where the file will be saved. Default is the current directory.  
        output_filename (str): Name of the output file (including extension). Default is "residuals_data.csv".  

    Returns:  
        str: Path to the saved file.  
    """  
    if not isinstance(y_test, np.ndarray) or not isinstance(y_pred, np.ndarray):  
        logging.error("y_test and y_pred must be provided as numpy arrays.")  
        return None  

    # Calculate residuals  
    residuals = y_test - y_pred  

    # Create a DataFrame  
    data = {  
        "y_test": y_test,  
        "y_pred": y_pred,  
        "residuals": residuals  
    }  
    df = pd.DataFrame(data)  

    # Ensure the output directory exists  
    os.makedirs(output_dir, exist_ok=True)  

    # Save to CSV file  
    output_path = os.path.join(output_dir, output_filename)  
    df.to_csv(output_path, index=False)  

    logging.info(f"Residual data saved to {output_path}")  


# ---------------------------------------------
# Helper function: Statistics
# --------------------------------------------- 
def metrics_for_predictions(
    y_test, 
    predictions
):  
    """  
    Calculate and return error metrics for model predictions.  

    Parameters:  
        y_test (torch.Tensor or numpy.ndarray): Ground truth labels.  
        predictions (torch.Tensor or numpy.ndarray): Model predictions.  

    Returns:  
        dict: A dictionary containing the following metrics:  
            - RMSE (Root Mean Squared Error)  
            - MAE (Mean Absolute Error)  
            - R² (R-squared Score)  
            - MAPE (Mean Absolute Percentage Error)  
    """  
    # Convert y_test to numpy array if it's a PyTorch tensor  
    if hasattr(y_test, 'numpy'):  # Check if y_test is a PyTorch tensor  
        y_test_np = y_test.numpy()  
    else:  
        y_test_np = y_test  # Assume y_test is already a numpy array  

    # Convert predictions to numpy array if it's a PyTorch tensor  
    if hasattr(predictions, 'numpy'):  # Check if predictions is a PyTorch tensor  
        predictions_np = predictions.numpy()  
    else:  
        predictions_np = predictions  # Assume predictions is already a numpy array  

    # Calculate Root Mean Squared Error (RMSE)  
    # RMSE measures the average magnitude of the errors between predictions and actual values.  
    rmse = np.sqrt(mean_squared_error(y_test_np, predictions_np))  

    # Calculate Mean Absolute Error (MAE)  
    # MAE measures the average absolute difference between predictions and actual values.  
    mae = mean_absolute_error(y_test_np, predictions_np)  

    # Calculate R-squared (R²) Score  
    # R² measures the proportion of variance in the dependent variable that is predictable from the independent variables.  
    r2 = r2_score(y_test_np, predictions_np)  

    # Calculate Mean Absolute Percentage Error (MAPE)  
    # MAPE measures the average percentage difference between predictions and actual values.  
    # Handle division by zero by replacing zeros in y_test_np with a small value (e.g., 1e-10).  
    y_test_np_safe = np.where(y_test_np == 0, 1e-10, y_test_np)  # Avoid division by zero  
    mape = np.mean(np.abs((y_test_np_safe - predictions_np) / y_test_np_safe)) * 100  

    # Cálculo do SMAPE  
    numerator = np.abs(y_test_np - predictions_np)  
    denominator = (np.abs(y_test_np) + np.abs(predictions_np)) / 2  
    smape = 100 * np.mean(numerator / denominator)  

    # Shapiro
    shapiro_value = shapiro(y_test_np - predictions_np)[1]

    # Store metrics in a dictionary for easy access and readability  
    metrics = {  
        "RMSE": float(rmse),  # Root Mean Squared Error  
        "MAE": float(mae),    # Mean Absolute Error  
        "R2": float(r2),      # R-squared Score  
        "MAPE": float(mape),   # Mean Absolute Percentage Error 
        "SMAPE": float(smape), # Symmetric Mean Absolute Percentage Error
        "Shapiro": float(shapiro_value)  # Shapiro-Wilk p-value (Residual Normality p-value)
    }  

    return metrics  


# ------------------------------------------------
# Helper function: Remove columns and Transform y
# ------------------------------------------------ 
def preprocess_dataset(
    data, 
    columns_to_exclude, 
    target_column, 
    transform="original"
):  
    """  
    Remove specified columns from a DataFrame and apply a transformation to one or more target columns.  

    Args:  
        data (pd.DataFrame): The input DataFrame.  
        columns_to_exclude (list): List of column names to exclude from the DataFrame.  
        target_column (str or list): The name of the column (single target) or list of column names (multiple targets) to transform.  
        transform (str): The transformation to apply to the target column(s).  
        Options:  
        - "original" (no transformation)  
        - "sqrt" (square root)  
        - "log" (natural logarithm with `log1p` for better precision)  

    Returns:  
        pd.DataFrame: A new DataFrame with the specified columns removed  
                      and the target column(s) transformed.  
    """ 
    # Validate inputs  
    if isinstance(target_column, str):  # Single target column  
        target_column = [target_column]  # Convert to a list for consistency  
    elif not isinstance(target_column, list):  # If not string or list, raise a ValueError  
        logging.error("'target_column' must be a string (single target) or a list (multiple targets).")  
        exit(1)

    # Ensure all target columns exist in the dataset  
    invalid_targets = [col for col in target_column if col not in data.columns]  
    if invalid_targets:  
        logging.error(f"The following target columns are not in the DataFrame: {invalid_targets}")  
        exit(1)

    if not isinstance(columns_to_exclude, list):
        logging.error("'columns_to_exclude' must be a list.") 
        exit(1)

    # Check for invalid columns in columns_to_exclude  
    invalid_columns = [col for col in columns_to_exclude if col not in data.columns]  
    if invalid_columns:  
        logging.error(f"The following columns to exclude are not in the DataFrame: {invalid_columns}")  
        exit(1)

    # Create a copy of the data to avoid modifying the original DataFrame  
    new_data = data.copy()  

    # Remove the specified columns  
    new_data = new_data.drop(columns=columns_to_exclude, errors='ignore')  

    # Apply the transformation to each target column  
    for col in target_column:  
        if transform == "original":  
            pass  # Do nothing, keep the original values  
        elif transform == "sqrt":  
            # Ensure no negative values to avoid errors  
            if (new_data[col] < 0).any():  
                logging.error(f"Cannot apply square root to negative values in '{col}'.")  
                exit(1)
            new_data[col] = np.sqrt(new_data[col])  
        elif transform == "log":  
            # Use log1p for better precision, ensure values are >= -1 to avoid errors  
            if (new_data[col] < -1).any():  
                logging.error(f"Cannot apply log1p transformation to values less than -1 in '{col}'.")  
                exit(1)
            new_data[col] = np.log1p(new_data[col])  
        else:  
            logging.error(f"Invalid transformation '{transform}'. Use 'original', 'sqrt', or 'log'.")  
            exit(1)

    return new_data  


# ---------------------------------------------
# Helper function: Revert transformation
# --------------------------------------------- 
def revert_transformation(
    data, 
    transform="original"
):  
    """  
    Revert a transformation applied to a NumPy array (sqrt or log).  

    Args:  
        data (np.ndarray): The NumPy array with transformed values.  
        transform (str): The type of transformation to revert.  
        Options:  
        - "original" (no transformation, return as-is)  
        - "sqrt" (revert square root, squares the values)  
        - "log" (revert log1p, applies expm1 to undo log1p)  

    Returns:  
        np.ndarray: A copy of the array with the transformation reverted.  
    """  
    # Create a copy of the original array to ensure it is not modified  
    data_copy = np.copy(data)  

    # Revert transformations based on the type  
    if transform == "original":  
        return data_copy  # No transformation applied, return as-is  
    elif transform == "sqrt":  
        return np.square(data_copy)  # Revert square root by squaring the values  
    elif transform == "log":  
        return np.expm1(data_copy)  # Revert log1p by applying expm1 (exp(x) - 1)  
    else:  
        logging.error(f"Invalid transformation '{transform}'. Use 'original', 'sqrt', or 'log'.")  
        exit(1)
