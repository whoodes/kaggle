import pandas as pd

# Exercise 1: Explore your data

# Setting so that all of the data columns are visible
pd.set_option('display.max_columns', None)

# Put the data into a pandas DataFrame
iowa_file_path = '~/datasets/housing-kaggle/train.csv'
iowa_data = pd.read_csv(iowa_file_path)

# Examination
print(iowa_data.describe())
