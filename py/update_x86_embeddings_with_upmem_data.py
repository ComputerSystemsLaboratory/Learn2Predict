#! /usr/bin/env python3

import os
import yaml as yl  
import numpy as np  

from absl import app  
from absl import flags  
from absl import logging  

from utils import parse_log_file  
from utils import upmem_log_files  
from utils import extract_upmem_log_info
from utils import extract_x86Histogram_embeddings_info  


# -------------------------  
# Define command-line flags  
# -------------------------  
FLAGS = flags.FLAGS  

flags.DEFINE_string(  
    'output_dir',  
    default='output',  
    help='Directory to store the dataset.'  
)  

flags.DEFINE_string(  
    'embeddings_dir',  
    default='embeddings',  
    help='x86 Histogram directory.'  
)  

flags.DEFINE_string(  
    'upmem_dir',  
    default='upmem',  
    help='UPMEM log directory.'  
)  

flags.DEFINE_list(  
    'benchmark',  
    default= [
                'BFS',
                'BS',
                'GEMV',
                'HST-L',
                'HST-S',
                'LiRnQ',
                'LiRQ',
                'LoRnQ',
                'LoRQ',
                'MLP',
                'RED',
                'SCAN-RSS',
                'SCAN-SSA',
                'SpMV',
                'TRNS',
                'TS',
                'VA'
            ],  
    help=(  
        'List of benchmarks to process. For example: '  
        'BFS,BS,GEMV,HST-L,...')  
)

flags.DEFINE_string(  
    'baseline',  
    default='tasklets_1_dpus_1_pass_0_param_0',  
    help='Baseline pattern.'  
)  


# -------------------------------------------------  
# Function to process a single benchmark  
# -------------------------------------------------  
def process_benchmark(
    benchmark, 
    upmem_dir, 
    embeddings_dir, 
    output_dir, 
    baseline_pattern
):  
    """  
    Process a single benchmark using UPMEM logs and embeddings.  

    Args:  
        benchmark (str): The benchmark name to process.  
        upmem_directory (str): The directory containing UPMEM logs.  
        embeddings_directory (str): The directory containing embeddings files.  
        output_directory (str): The directory to save the processed datasets.  
        baseline_pattern (str): The baseline pattern to locate the baseline file.  
    """  
    logging.info(f"Processing benchmark: {benchmark}")  

    # Locate the baseline file for this benchmark  
    baseline_file = os.path.join(upmem_dir, f"UPMEM_{benchmark}_{baseline_pattern}.log")  
    if not os.path.isfile(baseline_file):  
        logging.error(f"Baseline file does not exist: {baseline_file}")  
        return  

    # Parse the baseline UPMEM log file to retrieve the baseline timing 
    baseline = parse_log_file(baseline_file) 

    baseline = baseline["runtime"]

    # Locate all UPMEM files for the benchmark  
    upmem_files = upmem_log_files(upmem_dir, benchmark)  

    # Process each UPMEM log file  
    for upmem_file in upmem_files:  
        upmem_data = parse_log_file(os.path.join(upmem_dir, upmem_file), baseline)  

        if not upmem_data:  
            continue  

        # Extract x86Histogram embeddings pattern from the UPMEM file  
        pattern = extract_x86Histogram_embeddings_info(upmem_file)  

        # Find the corresponding histogram file  
        histogram_file = os.path.join(embeddings_dir, f"{benchmark}_{pattern}.yml")  
        if not os.path.isfile(histogram_file):  
            logging.warning(f"Histogram file not found: {histogram_file}")  
            continue  

        # Load histogram data  
        with open(histogram_file, "r") as yml_file:  
            histogram = yl.safe_load(yml_file)  

        if not histogram:  
            logging.warning(f"Histogram file is empty or invalid: {histogram_file}")  
            continue  

        # Combine UPMEM data and histogram into a single dataset  
        dataset = upmem_data | histogram  

        # Extract upmem log pattern from the UPMEM file  
        pattern = extract_upmem_log_info(upmem_file)

        # Save the combined dataset to the output directory 
        output_file = os.path.join(output_dir, f"{benchmark}_{pattern}.yml")  
        with open(output_file, "w") as yml_file:  
            yl.dump(dataset, yml_file)  

        logging.info(f"Dataset saved: {output_file}")  


# -------------------------------------------------  
# Main routine  
# -------------------------------------------------  
def main(
    argv
):  
    """  
    Main entry point for the script. Ensures input directorys are valid and processes all benchmarks.  
    """  
    # Validate UPMEM directory  
    if not os.path.isdir(FLAGS.upmem_dir):  
        logging.error(f"UPMEM directory does not exist: {FLAGS.upmem_dir}")  
        exit(1)  

    # Validate embeddings directory  
    if not os.path.isdir(FLAGS.embeddings_dir):  
        logging.error(f"Embeddings directory does not exist: {FLAGS.embeddings_dir}")  
        exit(1)  

    # Ensure output directory exists  
    os.makedirs(FLAGS.output_dir, exist_ok=True)  

    # Process each benchmark in the list  
    for benchmark in FLAGS.benchmark:  
        process_benchmark(  
            benchmark=benchmark,  
            upmem_dir=FLAGS.upmem_dir,  
            embeddings_dir=FLAGS.embeddings_dir,  
            output_dir=FLAGS.output_dir,  
            baseline_pattern=FLAGS.baseline  
        )  


if __name__ == '__main__':
    app.run(main)


