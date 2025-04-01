#! /usr/bin/env python3

import os
import yaml as yl
import numpy as np

from absl import app
from absl import flags
from absl import logging

from plot import plot_residuals
from plot import plot_training_history
from plot import plot_predictions_scatter


# -------------------------
# Define command-line flags
# -------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir',
    default='./output',
    help='Directory to save the the plots'
)

flags.DEFINE_string(
    'predictions',
    default='predictions_test.npz',
    help='Prediction filename'
)

flags.DEFINE_string(
    'history',
    default='history_test.yml',
    help='History filename'
)

flags.DEFINE_list(
    'target',
    default=[],
    help='The prediction (Speedup or Runtime)'
)

flags.DEFINE_list(
    'show_metrics',
    default=["MAPE", "R2", "Shapiro"],
    help='List of metrics to display on Predictions Plot'
)

flags.DEFINE_boolean(
    'plot_history',
    default=True,
    help='Plot training history'
)

flags.DEFINE_boolean(
    'plot_predictions',
    default=False,
    help='Plot predictions'
)

flags.DEFINE_boolean(
    'plot_residuals',
    default=False,
    help='Plot residuals'
)

# -------------------------------------------------
# Main routine
# -------------------------------------------------
def main(
    argv
):

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    if FLAGS.plot_history:
        if not os.path.isfile(FLAGS.history):
            logging.error(f"File does not exist: {FLAGS.history}")
            return

        with open(FLAGS.history, 'r', encoding='utf-8') as yml_file:
            history = yl.safe_load(yml_file)

        if not history:
            logging.error(f"History does not exist.")
            return

        plot_training_history(
            history,
            FLAGS.output_dir
        ) 

    if FLAGS.plot_predictions:
        if not os.path.isfile(FLAGS.predictions):
            logging.error(f"File does not exist: {FLAGS.predictions}")
            return

        predictions = np.load(FLAGS.predictions)
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']

        plot_predictions_scatter(
            y_test, 
            y_pred,
            FLAGS.target,
            FLAGS.show_metrics,
            FLAGS.output_dir
        )

    if FLAGS.plot_residuals:
        if not os.path.isfile(FLAGS.predictions):
            logging.error(f"File does not exist: {FLAGS.predictions}")
            return

        predictions = np.load(FLAGS.predictions)
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']

        plot_residuals(
            y_test, 
            y_pred, 
            FLAGS.output_dir
        )


if __name__ == '__main__':
    app.run(main)
