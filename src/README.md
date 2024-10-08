# Source code

This directory contains the main scripts/Jupyter notebooks developed for this project.

## Installing dependencies

This project requires Python. It was developed with Python 3.11 but may work with other versions (though this has not been tested).

In the root directory of this repository, run:
```
pip install -r requirements.txt
```

## Building the dataset

Running all cells in the `HBN_data-clean.ipynb` Jupyter notebook will create a file `hbn_fs_dataset.csv` that contains FreeSurfer 6.0.0 measures and other information (including sex, age, scan site ID) for a subset of participants in the Healthy Brain Network (HBN) dataset. This notebook takes around 15 minutes to run.

## "Splitting" the data

We add splitting columns to the dataset generated above to create partitions to be used by the server and each client. This is done by the `splitting_data_FLOWER.py` script (see below), and the name of the added column is determined by the options specified when calling the script.
```
usage: splitting_data_FLOWER.py [-h] [--n-splits-train N_SPLITS_TRAIN]
                                [--n-splits-test N_SPLITS_TEST]
                                [--random-state RANDOM_STATE] [--uneven] [--grouped_ages]
                                [--small_df_ratio SMALL_DF_RATIO]
                                data_csv out_csv

Update dataset with split indices

positional arguments:
  data_csv              Path to the dataset CSV file
  out_csv               Path to the output CSV file

options:
  -h, --help            show this help message and exit
  --n-splits-train N_SPLITS_TRAIN
                        Number of training splits (default: 3)
  --n-splits-test N_SPLITS_TEST
                        Number of splits for the test set (default: 5)
  --random-state RANDOM_STATE
                        Random seed (default: 42)
  --uneven              Use uneven splits
  --grouped_ages        Cannot be used in conjunction with --uneven
  --small_df_ratio SMALL_DF_RATIO
                        Integer to specify the ratio of the small df to rest of train
                        data.
```

The splits shown in our final presentation were obtained with the following parameters (with all other parameters set as default):
- `<N>_splits`, where `<N>` is an integer between 3 and 9 (inclusive): `--n-splits-train <N>`
- `3_splits_grouped`: `--n-splits-train 3 --grouped_ages`
- `3_splits_<M>_small`, where `<M>` is an integer between 4 and 9 (inclusive): `--n-splits-train 3 --uneven --small_df_ratio <M>`

## Running the federated learning network

The `server.py` script starts the server process, which will wait for a number of clients to connect to it then start the federated learning process. We set `--model LassoCV` to fit a lasso regression model. The `--split-col` argument is used to select one of the splits created in the previous step. `--y-col` can be left as the default value if predicting age using the HBN dataset generated by our Jupyter notebook.
```
usage: server.py [-h] [--model MODEL] [--min-clients MIN_CLIENTS] --data DATA
                 [--split-col SPLIT_COL] [--y-col Y_COL] [--path-out PATH_OUT]

Flower

options:
  -h, --help            show this help message and exit
  --model MODEL         Specifies the model used to fit
  --min-clients MIN_CLIENTS
                        Number of clients
  --data DATA           Path to data CSV file
  --split-col SPLIT_COL
                        Name of column used to split the data into train partitions and
                        test set
  --y-col Y_COL         Name of output variable column
  --path-out PATH_OUT   Name of output CSV file containing server/client metrics
```

The `client_newmodel.py` script starts a client process which will connect to the server and do local model training. We use a different `--partition-id` for each client so that they use a different subset of the dataset. The `--model` and `--split-col` arguments should be set to the same values as the server.

```
usage: client_newmodel.py [-h] --data DATA --partition-id PARTITION_ID [--model MODEL]
                          [--split-col SPLIT_COL] [--y-col Y_COL]

Flower

options:
  -h, --help            show this help message and exit
  --data DATA           Path to data CSV file
  --partition-id PARTITION_ID
                        Specifies the artificial data partition
  --model MODEL         Specifies the model used to fit
  --split-col SPLIT_COL
                        Name of column used to split the data into train partitions and
                        test set
  --y-col Y_COL         Name of output variable column
```

The `utils.py` file contains helper functions that are shared between `server.py` and `client_newmodel.py`.

## Training the model without federated learning

The `centralized_model.py` script trains a single `LassoCV` model on the entire training data. This was used to obtain the model performance metrics that we used as references when looking at the results of the federated learning models.
```
usage: centralized_model.py [-h] --data DATA [--split-col SPLIT_COL] [--y-col Y_COL]

options:
  -h, --help            show this help message and exit
  --data DATA           Path to data CSV file
  --split-col SPLIT_COL
                        Name of column used to split the data into train partitions and
                        test set
  --y-col Y_COL         Name of output variable column
```

## Other scripts

Finally, the `add_noise.py` contains (possibly incomplete/invalid) code to add noise to some of the splits. We did not end up using this code in the analysis we presented.

