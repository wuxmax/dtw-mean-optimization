import os
import glob
import csv
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
