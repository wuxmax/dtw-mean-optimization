# Comparison of Gradient-based Optimization Methods for theTime Series Sample Mean Problem in DTW Space

The program contained in this repository realizes the experiments described in the [respective paper](docs/paper.pdf).

Every operation described here is assumed to be executed from the root directory of this repository (the directory of this file).

## Run an experiment
To run an experiment, one simply has to execute the `main.py` in the `app` folder like this:

```
app/main.py
```

The `main.py` takes several command-line arguments, which can be seen by executing ```app/main.py --help```:

```
Run optimizing experiment for DTW mean computation.

positional arguments:
  CONFIG                the configuration to use in config folder

optional arguments:
  -h, --help            show this help message and exit
  -r PATH, --results PATH
                        path to store the results
  -d PATH, --datasets PATH
                        path of the datasets folders
```

If no `CONFIG` is given the `default.json` config is executed. If no results and dataset paths are given, the `datasets` and `results` folder relative to this directory are assumed.

## Configure experiment
To configure an experiment, a respective configuration file has to be placed in the `app/config` folder. The config file has to comply with the following format:

```
{
    "DATASETS" : ["Coffee"],
    "OPTIMIZERS" : {
        "sgld-120": {"method": "sgld", "n_coverage": 120, "batch_size": 3},
        "adam-100": {"method": "adam", "n_coverage": 100, "batch_size": 3},
        "ssg-100": {"method": "ssg", "n_coverage": 100, "batch_size": 3}
    },
    "NUM_ITERATIONS" : 3
}
```
For further explanation of the parameters, please refer to the paper.

To run the configuration in a file named `example.json`, the command `app/main.py example` or `app/main.py example.json` have to be executed.

## Deploy experiments on the daigpu3 VM
To run your experiments on the `daigpu3` VM (or any other system using Docker), follow these steps.

1. Log into the VM and clone or copy the repository.
2. `cd` into the repository.
3. If there are not already downloaded, you may use the `scripts/download_datasets.sh` script to load the *UCR Time Series Classification Archive* datasets to the correct directory.
4. Run `scripts/run_experiment.sh CONFIG` while replacing `CONFIG` with the correct config name. This scripts assumes the datasets and results directory to be subdirectories of this directory.

__NOTE__: This implementation highly profits from multiprocessing. So it advised to run it on a machine with a great number of processor cores (the `daigpu3` machine has 88 (v)cores).




## Result interpretation
The results are saved in the following format:
```
dataset, optimizer, iteration_id, variation, runtime
````
For an idea how to further analyze these results, you may refer to the [respective Jupyter Notebook](notebooks/explore_results.ipynb).
