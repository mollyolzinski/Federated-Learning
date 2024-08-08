# FEDERATED LEARNING WITH HEALTHY BRAIN NETWORK DATA: INTEGRATING FREESURFER METRICS AND AGE-BASED STRATIFICATION USING FLOWER
Federated Learning repository for Neurohackademy 2024

![download](https://github.com/user-attachments/assets/6574bff2-13c6-4b13-b17e-16008068521d)

## DESCRIPTION
Federated learning (FL) has become a promising machine learning method for conducting large-scale analyses across multiple institutions without the need to share data. With FL, data privacy and security are maintained, as the data never leave the institution; only encrypted model parameters are exchanged and aggregated. The aim of our project 

We first focus on collating FreeSurfer statistical data across multiple subjects, focusing on both cortical and subcortical brain region from the Healthy Brain Network (HBN). The HBN includes  MRI and clinical assessment data from 5,000 NYC area adolescents (ages 5-21), at four sites: RU, CBIC, CUNY, SI. For our specific sample we included 223 participants who had age and freesurfer deriviates available. Our script (client_newmodel.py) sets up a client for federated learning using the Flower framework package (see https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html for more details). It enables training and evaluation of machine learning models (Logistic Regression, Linear Regression, LassoCV) on different partitions of the HBN sample. The client communicates with a federated learning server to participate in model training and evaluation tasks over multiple rounds of federated learning, and saves the aggregated model weights after each round, in our case we use 3. 

Our Python script is designed to partition the HBN sample into training and test sets using stratified splitting and age binning as age was a continous variable. The process involves loading a dataset, creating stratification columns, splitting the data into multiple folds, and saving the resulting partitions to a CSV file.

Some of the Key Steps we have done include the following:

1) Data Loading and Preprocessing:
Load Data: Reads the dataset from a CSV file specified by the user.
Encode Categorical Variables: Maps categorical values (e.g., gender) to numerical values and drops irrelevant columns.
Age Binning: Creates a new column for age bins, categorizing ages into 2-year intervals.
2) Stratified Splitting:
Stratification Column: Concatenates values of specified stratification variables into a single column to ensure that splits maintain the distribution of these variables.
Training and Test Splits: Uses StratifiedKFold to create the specified number of training folds and a separate test set, ensuring that each fold has a representative distribution of the stratification variables.
3) Assign Split Indices:
Split Assignment: Assigns split indices to each partition and combines them into a single DataFrame, which includes a column indicating the split to which each row belongs.
4) Save Results:
Output CSV: Saves the partitioned dataset with split indices to a CSV file specified by the user.
5) Optional Visualization:
Visualization Code: Includes commented-out code for plotting the distribution of ages across splits and for different genders, which can be used for sanity checks and to ensure balanced splits.


Project url(s): (https://github.com/mollyolzinski/Federated-Learning.git) 

Contributors: [Michelle Wang](https://github.com/michellewang), [Emma Corley](https://github.com/emmajanecorley), [Eren Kafadar](https://github.com/kafadare), [Molly Olzinski](https://github.com/mollyolzinski), [Aoife Warren](https://github.com/AoifeWarren), [Audrey Weber](https://github.com/aweber7), [Maya Lakshman](https://github.com/mayalakshman) 


