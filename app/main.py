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

### CONFIG ###
# RESULTS_DIR = "../results"
# DATA_BASE_DIR = "../datasets/UCRArchive_2018/"
# DATASETS = ["Coffee"]
# OPTIMIZERS = {
#     "ssg-1": {'method': "ssg", 'n_epochs': 1, 'batch_size': 1},
#     "adam-1": {'method': "adam", 'n_epochs': 1, 'batch_size': 1}
# }
# NUM_ITERATIONS = 1
# RESULT_FORMAT = ["dataset", "optimizer", "iteration", "variation", "runtime"]
##############

def load_config():
    parser = argparse.ArgumentParser(description='Run optimizing experiment for DTW mean computation.')
    parser.add_argument('config', metavar='CONFIG', nargs='?', default="default",
        help='the configuration to use in config folder')
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

    return config


def run_experiment(config):
    c = config

    for dataset in c.DATASETS:

        logger.info(f"Starting experiment for dataset [ {dataset} ]")

        data = load_dataset(c.DATA_BASE_DIR, dataset)

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

                result_df = create_result_df(result, c.RESULT_FORMAT)
                results_file = save_result(result_df, c.RESULTS_DIR)

                logger.info(f"Saved latest results to [ {results_file} ]")

            logger.info(f"Finished experiment on [{dataset}] using [{opt_name}] for [{c.NUM_ITERATIONS}] iterations.")


if __name__=="__main__":
    config = load_config()
    run_experiment(config)
