from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Exercise 1: Explore your data

# Setting so that all of the data columns are visible
pd.set_option('display.max_columns', None)

# Put the data into a pandas DataFrame
iowa_file_path = '~/datasets/housing-kaggle/train.csv'
iowa_data = pd.read_csv(iowa_file_path)

# If data examination is so desired
iowa_data.describe()
iowa_data.columns

y = iowa_data.SalePrice

iowa_features = ['BedroomAbvGr', 'LotArea', 'GarageArea', 'GrLivArea', 'FullBath']
X = iowa_data[iowa_features]
actual = iowa_data['SalePrice']

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

print("Making predictions for the following five houses: ")
print(X.head())
print("The predictions: ")
print(iowa_model.predict(X.head()))
print("\nActual: ")
print(actual.head())
