#!/usr/bin/env python3

import os
import time
import logging
import multiprocessing as mp
from datetime import datetime
from numpy.random import SeedSequence, default_rng
from util import load_experiment_config, load_dataset, save_result
from optimizing import interface as opti

logging.basicConfig(level=logging.INFO, format=' %(name)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger("main")

### DEFAULT GENERAL CONFIG ###
general_config = {
    "RESULTS_DIR" : "./results",
    "DATA_BASE_DIR" : "./datasets/UCRArchive_2018/",
    "RESULT_FORMAT" : ("dataset", "optimizer", "iteration", "variation", "runtime")
}

def run_experiment(c):
    # timestamp to identify the experiment
    timestamp = datetime.now()

    # multiprocessing initialization
    manager = mp.Manager()
    queue = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() + 1)

    # calculate number of total iterations to show progress
    num_iterations_total = len(c['DATASETS']) * len(c['OPTIMIZERS']) * c['NUM_ITERATIONS']
    
    # start process which listens to the queue and writes new results to file
    result_writer = pool.apply_async(queue_listener, (queue, c, timestamp, num_iterations_total))

    # create independent random generator objects (streams) for every iteration
    seed_sequence = SeedSequence(12345)
    child_seeds = seed_sequence.spawn(num_iterations_total)
    random_streams = iter([default_rng(s) for s in child_seeds])

    # create all the workers which each compute one iteration
    iterations = []
    for dataset in c['DATASETS']:
        for opt_name, opt_params in c['OPTIMIZERS'].items():
            for iteration_idx in range(c['NUM_ITERATIONS']):
                rng = next(random_streams)                
                iteration = pool.apply_async(run_iteration, (c, dataset, iteration_idx, opt_name, opt_params, timestamp, queue, rng))
                iterations.append(iteration)

    # collect results from the workers through the pool result queue
    for iteration in iterations: 
        iteration.get()

    # now we are done, kill the listener
    queue.put('kill')
    pool.close()
    pool.join()


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
        logger.info(f"Total progress: [ {count_iterations_total} / {num_iterations_total} ] " \
                    f"--> Results saved to [ {results_file} ]")


def run_iteration(c, dataset, iteration_idx, opt_name, opt_params, timestamp, queue, rng):
    ''' run one iteration of the experiment '''

    logger.info(f"Starting iteration [ {iteration_idx + 1} ] using optimizer [ {opt_name} ] on dataset [ {dataset} ]")

    data = load_dataset(c['DATA_BASE_DIR'], dataset)

    runtime = None
    variation = None

    runtime, variation = opti.optimize_timed(data, **opt_params, random_stream=rng)

    iteration_id = "_".join([str(iteration_idx), str(hash(time.time()))])
    result = (dataset, opt_name, iteration_id, variation, runtime)

    logger.info(f"Finished iteration [ {iteration_idx + 1} ] using optimizer [ {opt_name} ] on dataset [ {dataset} ] " \
                f"in [ {runtime:.2f} ] seconds with [ {variation:.2f} ] variation")

    queue.put(result)


if __name__=="__main__":
    config = load_experiment_config(general_config)
    run_experiment(config)

