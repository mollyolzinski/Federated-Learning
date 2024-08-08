import flwr as fl
import utils
import numpy as np
import pandas as pd
import argparse
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss, mean_squared_error, r2_score 
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.svm import SVR
from typing import List, Union, Dict, Optional, Tuple

from flwr_datasets import FederatedDataset

dataset_full:pd.DataFrame = pd.read_csv("hbn_fs_data_split.csv")
dataset_full["sex"] = dataset_full["sex"].map({"M": 1, "F": 2})
dataset_full = dataset_full.drop (columns = ['subject_id', 
                                              'scan_site_id', 'ehq_total', 'commercial_use', 
                                              'full_pheno', 'expert_qc_score', 'xgb_qc_score', 
                                              'xgb_qsiprep_qc_score', 'dl_qc_score', 'site_variant', 
                                              'age_category', 'stratify_col'])
dataset_test:pd.DataFrame = dataset_full.loc[dataset_full['3_splits'] == -1]
y_test = dataset_test["age"]
X_test = dataset_test.drop(columns = ["age"])
dataset_train:pd.DataFrame = dataset_full.loc[dataset_full['3_splits'] == 1]
y_train = dataset_train["age"]
X_train = dataset_train.drop(columns = ["age"])

model = LinearRegression()

model.fit(X_train, y_train)

loss = mean_squared_error(y_test, model.predict(X_test))
accuracy = r2_score(y_test,model.predict(X_test))

print(loss, accuracy)