import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd

def load_dataset(data_base_dir, dataset):
    dataset_dir = os.path.join(os.path.abspath(data_base_dir), dataset)
    dfs = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".tsv"):
            file_path = os.path.join(dataset_dir, file)
            dfs.append(pd.read_csv(file_path, sep="\t", header=None))
    
    # merge train and test data
    df =  pd.concat(dfs)
    return df_to_np(df)

def df_to_np(df):
    np_array = df.to_numpy()
    return np.reshape(np_array, (np_array.shape[0], np_array.shape[1], 1))


def save_result(result_df, results_dir):
    results_dir = os.path.abspath(results_dir)
   
    # latest_results_file = get_latest_results_file(results_dir)
    
    # # if prior results already exists, merge data in one dataframe
    # results_df = None
    # if latest_results_file:
    #     latest_results_df = pd.read_csv(latest_results_file)
    #     results_df = pd.concat([latest_results_df, result_df])
    # if results_df is None:
    #     results_df = result_df
    
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # # results_filename = "results_" + timestamp + ".csv"
  
    results_filename = "results.csv"
    results_file = os.path.join(results_dir, results_filename)
    result_df.to_csv(results_file, index=False, mode='a+')
    # results_df.to_csv(results_file, index=False, mode='a+')
    return results_file


def get_latest_results_file(result_dir):    
    list_of_files = glob.glob(result_dir + '/*.csv') 
    latest_file = None
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def create_result_df(data_tuple, df_columns):
    return pd.DataFrame.from_records([data_tuple], columns=df_columns)
