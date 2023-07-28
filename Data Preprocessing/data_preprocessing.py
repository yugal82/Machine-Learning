# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# IMPORT DATASET
data = pd.read_csv('./Datasets/Data.csv')

# CONVERT DATASET INTO DEPENDENT VARIABLE AND FEATURE VAIRABLE
# X is the Feature variable matrix and Y is the Dependent variable matrix
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# TAKING CARE OF MISSING DATA AND NULL VALUES
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# ONE HOT ENCODING: Used to convert categorical variables to interger vairables
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

le = LabelEncoder()
y = le.fit_transform(y)

# Splitting dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Feature scaling.
# There might be some features which have very high values than other features that in turn may affect the result of the resulting model. To avoid this, we apply feature scaling to those features which accounts in model evaluation.
# Types for feature scaling: 1. Standardization (values between -3 and +3)
#                            2. Noramlization (values between 0, 1)

# Normalization is recemmended when data is normally distributed
# Generally Standardization works better all the time.

scaler = StandardScaler()
X_train[:, 3:] = scaler.fit_transform(X_train[:, 3:])
X_test[:, 3:] = scaler.transform(X_test[:, 3:])

print(X_train)
print(X_test)