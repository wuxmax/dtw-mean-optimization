import time
import logging
from util import *

logger = logging.getLogger("dtw_mean_opt_main")
logging.basicConfig(level=logging.INFO, format=' %(name)s :: %(levelname)s :: %(message)s')


### CONFIG ###
DATA_BASE_DIR = "/Users/Max/Documents/datasets/UCRArchive_2018/"
DATASETS = ["Coffee"]
OPTIMIZERS = {
    "ssg-1": {'func': "ssg", 'epochs': 1},
    # "ssg-2": {'func': "ssg", 'epochs': 2}
}
NUM_ITERATIONS = 1
RESULT_FORMAT = ["dataset", "optimizer", "iteration", "variation", "runtime"]
##############

def run_experiment():
    for dataset in DATASETS:

        logger.info(f"Starting experiment for dataset [ {dataset} ]")

        input_array = load_dataset(DATA_BASE_DIR, dataset)

        for opt_name, opt_params in OPTIMIZERS.items():

            logger.info(f"Using optimizer [ {opt_name} ]")
            
            for iteration_idx in range(NUM_ITERATIONS):

                logger.info(f"Running iteration [ {iteration_idx+1} / {NUM_ITERATIONS} ]")

                variation = None
                
                if opt_params['func'] == "ssg":

                    # timing needs to be adapted to avoid redundant calls
                    start_time = time.process_time()
                    
                    _, f_variations = ssg(input_array, n_epochs=opt_params['epochs'], return_f=True)
                    
                    end_time = time.process_time()
                    # end timing

                    print(f_variations)

                    variation = f_variations[-1]
                    runtime = end_time - start_time

                iteration_id = str(iteration_idx) + "_" + str(hash(time.time()))
                result = (dataset, opt_name, iteration_id, variation, runtime)

                result_df = create_result_df(result, RESULT_FORMAT)
                results_file = save_result(result_df)

                logger.info(f"Saved latest results to [ {results_file} ]")

            logger.info(f"Finished experiment on [{dataset}] using [{opt_name}] for [{NUM_ITERATIONS}] iterations")


if __name__=="__main__":
    run_experiment()