import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def split_dataframe(df : pd.DataFrame, stratify_vars: list[str], n_splits: int, random_state=42) -> list[pd.DataFrame]:
    
    
    # Step 2: Add stratify column
    # Creating a new column 'stratify_col' that concatenates the values of stratify_vars
    df['stratify_col'] = df[stratify_vars].astype(str).agg('-'.join, axis=1)
    
    # Step 3: Split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for _, test_index in skf.split(df, df['stratify_col']):
        splits.append(df.iloc[test_index])
    
    return splits

def partition_dataset(df: pd.DataFrame, n_splits: int,
                      n_splits_test: float, random_state=42, uneven:bool = False, 
                      small_df_ratio: int = None) -> pd.DataFrame:
    # CSV loaded 
    #df = pd.read_csv(path_data_csv)
    if uneven == True and small_df_ratio is not None:
        col_splits = f'{n_splits}_splits_{small_df_ratio}_small'
    elif uneven == False:
        col_splits = f'{n_splits}_splits'
    col_age_category = 'age_category'
    stratify_vars = [col_age_category, 'sex', 'scan_site_id']

    #find the min and max age and then bin variable by two year intervals 
    # define bin edges (2 year intervals)
    bins = range(0, int(df['age'].max()) + 2, 2) 
    labels = [f'{i}-{i+2}' for i in bins[:-1]]
    df[col_age_category] = pd.cut(df['age'], bins=bins, labels=labels, right=False) 

    # get the 20% test set
    train_test_splits = split_dataframe(df=df, stratify_vars=stratify_vars, n_splits=n_splits_test, random_state=random_state)
    df_test = train_test_splits[0]
    df_test[col_splits] = -1

    df_train = pd.concat(train_test_splits[1:])
    # Sort DataFrame by age
 #   df_train = df_train.sort_values(by='age')

    # Split the DataFrame into thirds
    # third = len(df_train) // 3
    # df_train['three_splits_unbalanced'] = np.where(df_train.index < third, 1, 
    #                    np.where(df_train.index < 2 * third, 2, 3))
    if uneven == True and small_df_ratio is not None:
        splits_small = split_dataframe(df=df_train, stratify_vars=stratify_vars, n_splits=small_df_ratio, random_state=random_state)
        df_small = splits_small[n_splits-1]
        df_small[col_splits] = n_splits-1
        df_rest = pd.concat(splits_small[1:])
        splits = split_dataframe(df=df_rest, stratify_vars=stratify_vars, n_splits=(n_splits-1), random_state=random_state)
        for i in range(n_splits-1):
            splits[i][col_splits] = int(i)
        splits = splits + [df_small]
        
    elif uneven == False:
        splits = split_dataframe(df=df_train, stratify_vars=stratify_vars, n_splits=n_splits, random_state=random_state)
        # for i, split in enumerate(splits):
        #     print(f"Split {i+1}:\n", split['stratify_col'].head()) 
        for i in range(n_splits):
            splits[i][col_splits] = int(i)
    
    #function returns dataframe split into datasets concatanated with the test dataframe
    return pd.concat(splits + [df_test])

    #Create uneven data splits
    

    #sanity check for distribution 

    # Plotting the original age variable 

    # plt.figure(figsize=(12, 6))
    # for i, split in enumerate(splits):
    #     sns.histplot(split['age'], bins=len(labels), kde=False, label=f'Split {i+1}', alpha=0.5)

    # plt.title('Age Category Distribution in Each Split')
    # plt.xlabel('Age')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

    #new 

    # # Plotting
    # n_splits = len(splits)
    # genders = df['sex'].unique()
    # genders = [str(gender) for gender in genders if pd.notna(gender)]  # Ensure only valid strings
    # n_genders = len(genders)  # Get the number of unique genders

    # # Create a figure with subplots
    # fig, axes = plt.subplots(nrows=n_genders, ncols=n_splits, figsize=(20, 10), sharey=True)

    # # Plot each split for each gender
    # for i, split in enumerate(splits):
    #     for j, gender in enumerate(genders):
    #         ax = axes[j, i]  # Use j for gender index
    #         sns.histplot(split[split['sex'] == gender]['age'], bins=20, ax=ax, kde=True, stat='density', alpha=0.5)
    #         ax.set_title(f'Split {i + 1} - {str(gender).capitalize()}')  # Ensure gender is a string
    #         ax.set_xlabel('Age')
    #         ax.set_ylabel('Count')
    #         ax.tick_params(axis='x', rotation=45)

    # # Adjust layout
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':

    DEFAULT_N_SPLITS_TRAIN = 3
    DEFAULT_N_SPLITS_TEST = 5
    DEFAULT_RANDOM_STATE = 42
    
    parser = argparse.ArgumentParser(description="Update dataset with split indices")
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
        "--n-splits-train",
        type=int,
        default=DEFAULT_N_SPLITS_TRAIN,
        help=f"Number of training splits (default: {DEFAULT_N_SPLITS_TRAIN})",
    )
    parser.add_argument(
        "--n-splits-test",
        type=float,
        default=DEFAULT_N_SPLITS_TEST,
        help=f"Number of splits for the test set (default: {DEFAULT_N_SPLITS_TEST})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed (default: {DEFAULT_RANDOM_STATE})",
    )
    parser.add_argument(
        "--uneven",
        type=bool,
        default=False,
        help=f"Boolean to specify whether split should be uneven",
    )
    parser.add_argument(
        "--small_df_ratio",
        type=int,
        default=None,
        help=f"Integer to specify the ratio of the small df to rest of train data.",
    )
    args = parser.parse_args()

    path_data_csv = args.data_csv
    path_out = args.out_csv
    n_splits_train = args.n_splits_train
    n_splits_test = args.n_splits_test
    random_state = args.random_state
    uneven = args.uneven
    small_df_ratio = args.small_df_ratio
    if uneven == True and small_df_ratio is not None:
        col_splits = f'{n_splits_train}_splits_{small_df_ratio}_small'
    elif uneven == False:
        col_splits = f'{n_splits_train}_splits'
    
    df = pd.read_csv(path_data_csv)
    df_split = partition_dataset(df, n_splits_train, n_splits_test, random_state, uneven, small_df_ratio)
    print(df_split[col_splits].value_counts())
    df_split.to_csv(path_out, index=False)
