import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def split_dataframe(df : pd.DataFrame, stratify_vars: list[str], n_splits: int) -> list[pd.DataFrame]:
    
    
    # Step 2: Add stratify column
    # Creating a new column 'stratify_col' that concatenates the values of stratify_vars
    df['stratify_col'] = df[stratify_vars].astype(str).agg('-'.join, axis=1)
    
    # Step 3: Split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    
    for _, test_index in skf.split(df, df['stratify_col']):
        splits.append(df.iloc[test_index])
    
    return splits

# CSV loaded 
path_csv= '/Users/audreyweber/Documents/Federated-Learning/input/participants.csv'
df = pd.read_csv(path_csv)
#find the min and max age and then bin variable by two year intervals 

# define bin edges (2 year intervals)

bins = range(0, int(df['age'].max()) + 2, 2) 
labels = [f'{i}-{i+2}' for i in bins[:-1]]

df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels, right=False) 

splits = split_dataframe(df, ['age_category', 'sex', 'scan_site_id'], 5)
for i, split in enumerate(splits):
    print(f"Split {i+1}:\n", split['stratify_col'].head())

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