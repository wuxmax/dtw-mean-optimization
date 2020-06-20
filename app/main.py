#!/usr/bin/env python3

import os
import time
import logging
import argparse
import json
from collections import namedtuple    
from util import *
from optimizing import interface as opti

logging.basicConfig(level=logging.INFO, format=' %(name)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger("dtw_mean_opt_main")

### GENERAL DEFAULT CONFIG ###
RESULTS_DIR = "./results"
# DATA_BASE_DIR = "/Users/Max/Documents/datasets/UCRArchive_2018/"
DATA_BASE_DIR = "./datasets/UCRArchive_2018/"
RESULT_FORMAT = ["dataset", "optimizer", "iteration", "variation", "runtime"]


def load_experiment_config():
    parser = argparse.ArgumentParser(description='Run optimizing experiment for DTW mean computation.')
    parser.add_argument('config', metavar='CONFIG', nargs='?', default="default",
        help='the configuration to use in config folder')
    parser.add_argument('config', metavar='CONFIG', nargs='?', default="default",
        help='the configuration to use in config folder')
    parser.add_argument('--results', metavar='PATH', dest='results_path',
                    help='path to store the results')
    parser.add_argument('--dataset', metavar='PATH', dest='datasets_path',
                    help='path of the datasets folders')

    args = parser.parse_args()
    config_filename = args.config
    
    if not config_filename.endswith(".json"):
        config_filename += ".json"

    this_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(this_dir, 'config', config_filename)
    
    try:
        with open(config_file) as f:
            config_dict = json.load(f)
    except Exception as e:
        logger.exception("Could not load config file" + str(e))

    logger.info(f"Loaded configuration [ {config_filename} ]")
    
    Config = namedtuple('Config', config_dict.keys()) 
    config = Config(**config_dict)

    # change general config if given
    if args.results_path: RESULTS_DIR = args.results_path 
    if args.datasets_path: DATA_BASE_DIR = args.datasets_path 

    return config


def run_experiment(config):
    c = config

    for dataset in c.DATASETS:

        logger.info(f"Starting experiment for dataset [ {dataset} ]")

        data = load_dataset(DATA_BASE_DIR, dataset)

        for opt_name, opt_params in c.OPTIMIZERS.items():

            logger.info(f"Using optimizer [ {opt_name} ]")

            for iteration_idx in range(c.NUM_ITERATIONS):

                logger.info(f"Running iteration [ {iteration_idx+1} / {c.NUM_ITERATIONS} ]")

                runtime = None
                variation = None

                #  optimize(X, method=None, n_epochs=None, batch_size=1, init_sequence=None, return_z=False)
                runtime, variation = opti.optimize_timed(data, **opt_params)

                iteration_id = str(iteration_idx) + "_" + str(hash(time.time()))
                result = (dataset, opt_name, iteration_id, variation, runtime)

                result_df = create_result_df(result, RESULT_FORMAT)
                results_file = save_result(result_df, RESULTS_DIR)

                logger.info(f"Saved latest results to [ {results_file} ]")

            logger.info(f"Finished experiment on [{dataset}] using [{opt_name}] for [{c.NUM_ITERATIONS}] iterations.")
            
            num_iterations_total = len(c.DATASETS) * len(c.OPTIMIZERS) * c.NUM_ITERATIONS
            idx_iterations_total = (c.DATASETS.index(dataset) * len(c.OPTIMIZERS) + list(c.OPTIMIZERS.keys()).index(opt_name)) \
                * c.NUM_ITERATIONS + iteration_idx + 1
            logger.info(f"### Total progress: [ {idx_iterations_total} / {num_iterations_total} ] iterations ###")


if __name__=="__main__":
    config = load_experiment_config()
    run_experiment(config)
