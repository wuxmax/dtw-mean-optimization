import os
import glob
import csv
import json
import numpy as np
import pandas as pd
import argparse
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)

def load_experiment_config(general_config):
    parser = argparse.ArgumentParser(description='Run optimizing experiment for DTW mean computation.')
    parser.add_argument('config', metavar='CONFIG', nargs='?', default="default",
        help='the configuration to use in config folder')
    parser.add_argument('-r','--results', metavar='PATH', dest='results_path',
                    help='path to store the results')
    parser.add_argument('-d', '--datasets', metavar='PATH', dest='datasets_path',
                    help='path of the datasets folders')

    args = parser.parse_args()
    config_name = args.config

    config_filesuffix = ".json"    
    if config_name.endswith(config_filesuffix):
        config_filename = config_name
        config_name = config_name[:-len(config_filesuffix)]
    else:
        config_filename = config_name + config_filesuffix

    this_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(this_dir, 'config', config_filename)
    
    # try to load json config file into exp_config
    try:
        with open(config_file) as f:
            exp_config = json.load(f)
    except Exception as e:
        logger.exception("Could not load config file: " + str(e))
        exit(1)

    exp_config['NAME'] = config_name
    logger.info(f"Loaded configuration [ {config_name} ]")

    # change general config if given
    if args.results_path: general_config['RESULTS_DIR'] = args.results_path         
    if args.datasets_path: general_config['DATA_BASE_DIR'] = args.datasets_path 

    config = {**general_config, **exp_config}

    return config

def load_dataset(data_base_dir, dataset):
    dataset_dir = os.path.join(os.path.abspath(data_base_dir), dataset)
    dfs = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".tsv"):
            file_path = os.path.join(dataset_dir, file)
            df = pd.read_csv(file_path, sep="\t", header=None)
            # exclude first column (class label)
            df.drop(columns=[0], inplace=True)
            dfs.append(df)
    
    # merge train and test data
    if len(dfs) > 1:
        df =  pd.concat(dfs)
    
    return df_to_np(df)

def df_to_np(df):
    np_array = df.to_numpy()
    return np.reshape(np_array, (np_array.shape[0], np_array.shape[1], 1))


def save_result(result, results_dir, result_format, exp_name, exp_timestamp):
    results_dir = os.path.abspath(results_dir)
    results_filename = "_".join(["results", exp_name, exp_timestamp.strftime("%d-%m-%Y_%H-%M-%S")]) + ".csv"
    results_file = os.path.join(results_dir, results_filename)

    file_exists = os.path.exists(results_file)
    
    with open(results_file,'a+') as out:
        csv_out=csv.writer(out)
        # write header if file does not exist
        if not file_exists:
            csv_out.writerow(result_format)
        csv_out.writerow(result)

    return results_file


# def get_latest_results_file(result_dir):    
#     list_of_files = glob.glob(result_dir + '/*.csv') 
#     latest_file = None
#     if list_of_files:
#         latest_file = max(list_of_files, key=os.path.getctime)
#     return latest_file


# def create_result_df(data_tuple, df_columns):
#     return pd.DataFrame.from_records([data_tuple], columns=df_columns)
