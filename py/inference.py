#! /usr/bin/env python3

import os
import yaml as yl
import numpy as np
import pandas as pd

from absl import app
from absl import flags
from absl import logging

from plot import plot_residuals
from plot import plot_training_history
from plot import plot_predictions_scatter

from model import predict_with_pretrained_model

from utils import preprocess_dataset
from utils import revert_transformation
from utils import metrics_for_predictions


# -------------------------
# Define command-line flags
# -------------------------
FLAGS = flags.FLAGS

flags.DEFINE_list(
    'inference_benchmarks',
    default=[
            'BFS',
            'BS',
            'GEMV',
            'HST-L',
            'HST-S',
            'MLP',
            'RED',
            'SCAN-RSS',
            'SCAN-SSA',
            'SpMV',
            'TRNS',
            'TS',
            'VA'
    ],
    help='Test benchmarks'
)

flags.DEFINE_enum(
    'transformation',
    default='log',
    enum_values=[
                'original',
                'sqrt',
                'log'
    ],
    help='Apply a transformation to the target column'
)

flags.DEFINE_string(
    'dataset_dir',
    default='./dataset_dir',
    help='Dataset directory (CSV files)'
)

flags.DEFINE_list(
    'exclude_columns',
    default=[],
    help='List of columns to exclude from features'
)

flags.DEFINE_list(
    'target_column',
    default=None,
    help='Column to use as the target (y)'
)

flags.DEFINE_string(
    'model_dir',
    default='./model',
    help='Directory to save the best model'
)

flags.DEFINE_string(
    'output_dir',
    default='./output',
    help='Directory to save the predictions, metrics, and history'
)

flags.DEFINE_string(
    'model_name',
    default='best_model.pth',
    help='Name of the saved model file'
)

flags.DEFINE_integer(
    'patience',
    default=20,
    help='Patience for early stopping'
)

flags.DEFINE_integer(
    'num_epochs',
    default=100,
    help='Number of epochs for fine-tuning the model'
)

flags.DEFINE_boolean(
    'use_patience',
    default=False,
    help='Apply early stopping if True'
)

flags.DEFINE_integer(
    'freeze_up_to',
    default=6,
    help='Number of layers to freeze'
)

flags.DEFINE_boolean(
    'fine_tuning',
    default=True,
    help='Enable fine-tuning the model'
)

flags.DEFINE_integer(
    'random_state',
    default=42,
    help='Random state for reproducibility'
)

flags.DEFINE_float(
    'fine_tuning_size',
    default=0.1,
    help='Proportion of the data for fine-tuning the model'
)

flags.DEFINE_float(
    'val_size',
    default=0.1,
    help='Proportion of the data for validation'
)

flags.DEFINE_float(
    'learning_rate_reduction',
    default=0.1,
    help='Learing rate reduction'
)

flags.DEFINE_string(
    'predictions_filename',
    default='predictions_test.npz',
    help='Prediction filename'
)

flags.DEFINE_string(
    'metrics_filename',
    default='metrics_test.yml',
    help='Metrics filename'
)

flags.DEFINE_string(
    'history_filename',
    default='history_test.yml',
    help='Fine tuning history filename'
)

flags.DEFINE_string(
    'elapsed_filename',
    default='elapsed_time.yml',
    help='Elapsed time filename'
)

flags.DEFINE_integer(
    'hidden_size',
    default=256,
    help='Number of neurons in the hidden layer'
)

flags.DEFINE_integer(
    'batch_size',
    default=16,
    help='Size of each batch for training and validation'
)

flags.DEFINE_float(
    'learning_rate',
    default=0.0001,
    help='Learning rate for the optimizer'
)

flags.DEFINE_float(
    'dropout_rate',
    default=0.5,
    help='Dropout rate for regularization'
)

flags.DEFINE_integer(
    'num_layers',
    12,
    'Number of hidden layers'
)

# Mark required flags  
flags.mark_flag_as_required('target_column')  


# -------------------------------------------------
# Main routine
# -------------------------------------------------
def main(
    argv
):
    
    if not os.path.isdir(FLAGS.dataset_dir):
        logging.error(f"Directory does not exist: {FLAGS.dataset_dir}")
        exit(1)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    dataframes = []  # List to store individual DataFrames  
    for benchmark in FLAGS.inference_benchmarks:  
        # Validate file existence and extension 
        file_name = os.path.join(FLAGS.dataset_dir, f"{benchmark}.csv")
        if not os.path.exists(file_name):  
            logging.error(f"File '{file_name}' does not exist.")  
            exit(1)
        if not file_name.endswith('.csv'):  
            logging.error(f"File '{file_name}' is not a CSV file.")  
            exit(1)

        # Load the CSV file into a DataFrame  
        try:  
            df = pd.read_csv(file_name)  
            dataframes.append(df)  
        except Exception as e:  
            logging.error(f"Failed to load CSV file '{file_name}': {e}")  
            exit(1)

    # Concatenate all DataFrames into one  
    data = pd.concat(dataframes, ignore_index=True)

    target_column = FLAGS.target_column[0] if len(FLAGS.target_column) == 1 else FLAGS.target_column
    
    new_data = preprocess_dataset(data, FLAGS.exclude_columns, FLAGS.target_column, FLAGS.transformation)

    y_test, predictions, history, elapsed_time = predict_with_pretrained_model(  
                                                    new_data,
                                                    target_column=target_column,
                                                    hidden_size=FLAGS.hidden_size,  
                                                    output_size=len(FLAGS.target_column),
                                                    learning_rate=FLAGS.learning_rate*FLAGS.learning_rate_reduction,
                                                    dropout_rate=FLAGS.dropout_rate,
                                                    batch_size=FLAGS.batch_size,
                                                    use_patience=FLAGS.use_patience,
                                                    patience=FLAGS.patience,
                                                    num_epochs=FLAGS.num_epochs,
                                                    test_size=1.0-FLAGS.fine_tuning_size,  
                                                    val_size=FLAGS.val_size,
                                                    random_state=FLAGS.random_state,
                                                    model_dir=FLAGS.model_dir,  
                                                    model_name=FLAGS.model_name,
                                                    save_model=False,
                                                    fine_tuning=FLAGS.fine_tuning,
                                                    freeze_up_to=FLAGS.freeze_up_to
    )
    
    y_test_revert = revert_transformation(y_test, FLAGS.transformation)
    predictions_revert = revert_transformation(predictions, FLAGS.transformation)

    metrics = metrics_for_predictions(y_test_revert, predictions_revert)

    output_npz = os.path.join(FLAGS.output_dir, FLAGS.predictions_filename)
    np.savez_compressed(
        output_npz,
        y_test=y_test_revert,
        y_pred=predictions_revert
    )
    logging.info(f"Predictions saved to {output_npz}")
 
    metrics_yml = os.path.join(FLAGS.output_dir, FLAGS.metrics_filename)
    with open(metrics_yml, 'w', encoding='utf-8') as yml_file:
        yl.dump(metrics, yml_file, default_flow_style=False, sort_keys=True)
    logging.info(f"Metrics saved to {metrics_yml}")

    history_yml = os.path.join(FLAGS.output_dir, FLAGS.history_filename)
    with open(history_yml, 'w', encoding='utf-8') as yml_file:
        yl.dump(history, yml_file, default_flow_style=False, sort_keys=True)
    logging.info(f"History saved to {history_yml}")

    elapsed_yml = os.path.join(FLAGS.output_dir, FLAGS.elapsed_filename)
    with open(elapsed_yml, 'w', encoding='utf-8') as yml_file:
        yl.dump(elapsed_time, yml_file, default_flow_style=False, sort_keys=True)
    logging.info(f"Elapsed time saved to {elapsed_yml}")


if __name__ == '__main__':
    app.run(main)
