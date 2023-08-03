# This regression file is from a YouTube tutorial.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('./Datasets/homeprices.csv')
# print(data.head(3))

# to handle null/missing values
data['bedrooms'] = data['bedrooms'].fillna(value=data['bedrooms'].median())

X = data[['area', 'bedrooms', 'age']]
y = data[['price']]

regressor = LinearRegression()
regressor.fit(X, y)

# print(regressor.coef_)
# print(regressor.intercept_)

print(regressor.predict([[3000, 3, 40]]))