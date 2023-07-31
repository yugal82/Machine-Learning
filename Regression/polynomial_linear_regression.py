import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('./Datasets/Position_Salaries.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# __________________________________ LINEAR REGRESSION _______________________________________________________
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X, y, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

print(regressor.predict([[6]]))

# __________________________________ POLYNOMIAL REGRESSION _______________________________________________________
poly_regressor = PolynomialFeatures(degree = 2)
X_poly = poly_regressor.fit_transform(X)
lin_regressor = LinearRegression()
lin_regressor.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regressor.predict(X_poly), color = 'blue')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

print(lin_regressor.predict(poly_regressor.fit_transform([[6.5]])))