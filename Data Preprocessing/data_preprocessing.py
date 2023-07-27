# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
print(y)
