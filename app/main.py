#!/usr/bin/env python3

import os
import time
import logging
import multiprocessing as mp
from datetime import datetime
from util import load_experiment_config, load_dataset, save_result
from optimizing import interface as opti

logging.basicConfig(level=logging.INFO, format=' %(name)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger("main")

### DEFAULT GENERAL  CONFIG ###
general_config = {
    "RESULTS_DIR" : "./results",
    "DATA_BASE_DIR" : "./datasets/UCRArchive_2018/",
    "RESULT_FORMAT" : ("dataset", "optimizer", "iteration", "variation", "runtime")
}

def queue_listener(queue, c, timestamp, num_iterations_total):
    '''listens for messages on the queue, writes to file. '''
    
    count_iterations_total = 0
    while True:
        msg = queue.get()
        if msg == 'kill':
            break

        # if not, msg is latest result
        count_iterations_total += 1
        results_file = save_result(msg, c['RESULTS_DIR'], c['RESULT_FORMAT'], c['NAME'], timestamp)
        logger.info(f"Total progress: [ {count_iterations_total} / {num_iterations_total} " \
                    f"--> Results saved to [ {results_file} ]")


def run_experiment(c):
    timestamp = datetime.now()

    manager = mp.Manager()
    queue = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() + 1)

    num_iterations_total = len(c['DATASETS']) * len(c['OPTIMIZERS']) * c['NUM_ITERATIONS']
    result_writer = pool.apply_async(queue_listener, (queue, c, timestamp, num_iterations_total))

    iterations = []
    for dataset in c['DATASETS']:
        for opt_name, opt_params in c['OPTIMIZERS'].items():
            for iteration_idx in range(c['NUM_ITERATIONS']):
                
                pbar_position = len(iterations)

                iteration = pool.apply_async(run_iteration, (c, dataset, iteration_idx, opt_name, opt_params, timestamp, queue))
                iterations.append(iteration)

    # collect results from the workers through the pool result queue
    for iteration in iterations: 
        iteration.get()

    # now we are done, kill the listener
    queue.put('kill')
    pool.close()
    pool.join()


def run_iteration(c, dataset, iteration_idx, opt_name, opt_params, timestamp, queue):
    logger.info(f"Starting iteration [ {iteration_idx + 1} ] using optimizer [ {opt_name} ] on dataset [ {dataset} ]")

    data = load_dataset(c['DATA_BASE_DIR'], dataset)

    runtime = None
    variation = None

    #  optimize(X, method=None, n_epochs=None, batch_size=1, init_sequence=None, return_z=False)
    runtime, variation = opti.optimize_timed(data, **opt_params)

    iteration_id = "_".join([str(iteration_idx), str(hash(time.time()))])
    result = (dataset, opt_name, iteration_id, variation, runtime)

    logger.info(f"Finished iteration [ {iteration_idx + 1} ] using optimizer [ {opt_name} ] on dataset [ {dataset} ] " \
                f"in [ {runtime:.2f} ] seconds with [ {variation:.2f} ] variation")

    queue.put(result)


if __name__=="__main__":
    config = load_experiment_config(general_config)
    run_experiment(config)


# def run_experiment(c):

#     timestamp = datetime.now()

#     for dataset in c.DATASETS:

#         logger.info(f"Starting experiment for dataset [ {dataset} ]")

#         data = load_dataset(c.DATA_BASE_DIR, dataset)

#         logger.info(f"Dataset size: [ {data.shape[0]} ]")

#         for opt_name, opt_params in c.OPTIMIZERS.items():

#             logger.info(f"Using optimizer [ {opt_name} ]")

#             logger.info(f"Min. visited samples: [ {opt_params['n_coverage']} ]")

#             logger.info(f"Batch size: [ {opt_params['batch_size']} ]")

#             for iteration_idx in range(c.NUM_ITERATIONS):
              
                # runtime = None
                # variation = None

                # #  optimize(X, method=None, n_epochs=None, batch_size=1, init_sequence=None, return_z=False)
                # runtime, variation = opti.optimize_timed(data, **opt_params)

                # iteration_id = str(iteration_idx) + "_" + str(hash(time.time()))
                # result = (dataset, opt_name, iteration_id, variation, runtime)

                # results_file = save_result(result, c.RESULTS_DIR, c.RESULT_FORMAT, c.NAME, timestamp)

                # logger.info(f"Saved latest results to [ {results_file} ]")

#             logger.info(f"Finished experiment on [{dataset}] using [{opt_name}] for [{c.NUM_ITERATIONS}] iterations")
            
#             num_iterations_total = len(c.DATASETS) * len(c.OPTIMIZERS) * c.NUM_ITERATIONS
#             idx_iterations_total = (c.DATASETS.index(dataset) * len(c.OPTIMIZERS) + list(c.OPTIMIZERS.keys()).index(opt_name)) \
#                 * c.NUM_ITERATIONS + iteration_idx + 1
#             logger.info(f"### Total progress: [ {idx_iterations_total} / {num_iterations_total} ] iterations ###")
