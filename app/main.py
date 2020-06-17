import time
import logging
from util import *
from optimizing import interface as opti

logging.basicConfig(level=logging.INFO, format=' %(name)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger("dtw_mean_opt_main")

### CONFIG ###
DATA_BASE_DIR = "/Users/Max/Documents/datasets/UCRArchive_2018/"
RESULTS_DIR = "../results"
DATASETS = ["Coffee"]
OPTIMIZERS = {
    "ssg-1": {'method': "ssg", 'n_epochs': 1, 'batch_size': 1},
    # "ssg-2": {'func': "ssg", 'n_epochs': 2}
}
NUM_ITERATIONS = 1
RESULT_FORMAT = ["dataset", "optimizer", "iteration", "variation", "runtime"]
##############

def run_experiment():
    for dataset in DATASETS:

        logger.info(f"Starting experiment for dataset [ {dataset} ]")

        data = load_dataset(DATA_BASE_DIR, dataset)

        for opt_name, opt_params in OPTIMIZERS.items():

            logger.info(f"Using optimizer [ {opt_name} ]")
            
            for iteration_idx in range(NUM_ITERATIONS):

                logger.info(f"Running iteration [ {iteration_idx+1} / {NUM_ITERATIONS} ]")

                runtime = None
                variation = None

                #  optimize(X, method=None, n_epochs=None, batch_size=1, init_sequence=None, return_z=False)
                runtime, variation = opti.optimize_timed(data, **opt_params)

                iteration_id = str(iteration_idx) + "_" + str(hash(time.time()))
                result = (dataset, opt_name, iteration_id, variation, runtime)

                result_df = create_result_df(result, RESULT_FORMAT)
                results_file = save_result(result_df, RESULTS_DIR)

                logger.info(f"Saved latest results to [ {results_file} ]")

            logger.info(f"Finished experiment on [{dataset}] using [{opt_name}] for [{NUM_ITERATIONS}] iterations.")


if __name__=="__main__":
    run_experiment()