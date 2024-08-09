import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def add_noise(data:pd.Series, noise_level:float = 1.0):
    noise = np.random.normal(loc=0, scale=noise_level * data.mean(), size=len(data))
    print(noise)
    return data + noise

if __name__ == '__main__':

    DEFAULT_NOISE_LEVEL = 1.0
    
    parser = argparse.ArgumentParser(description="Add noise to a specified split of the dataset")
    parser.add_argument(
        "data_csv",
        type=str,
        help="Path to the dataset CSV file",
    )
    parser.add_argument(
        'out_csv',
        type=str,
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=DEFAULT_NOISE_LEVEL,
        help=f"Noise level (default: {DEFAULT_NOISE_LEVEL})",
    )
    parser.add_argument(
        "--split-column-name",
        type=str,
        help=f"Name of column to specify splits",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        help=f"Index for partition to add noise to",
    )
    args = parser.parse_args()

    path_data_csv = args.data_csv
    path_out = args.out_csv
    noise_level = args.noise_level
    split_column_name = args.split_column_name
    partition_id = args.partition_id

    df = pd.read_csv(path_data_csv)

    columns_with_splits = [col for col in df.columns if 'splits' in col]
    exclude_columns = ['subject_id', 'scan_site_id', 'ehq_total', 'commercial_use', 
                            'full_pheno', 'expert_qc_score', 'xgb_qc_score', 
                             'xgb_qsiprep_qc_score', 'dl_qc_score', 'site_variant',
                             'age_category', 'stratify_col', 'age', 'sex'] + columns_with_splits
    columns_add_noise = [col for col in df.columns if col not in exclude_columns]
    #print(columns_add_noise)
    #run through the feature columns to add noise
    for column in columns_add_noise:
        df.loc[df[split_column_name] == partition_id, column] = add_noise(
        df.loc[df[split_column_name] == partition_id, column],
        noise_level)
    
    result = df.groupby(split_column_name)["Cortex"].agg(['mean', 'std'])
    print(result)
    df.to_csv(path_out, index=False)

   
