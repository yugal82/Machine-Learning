# This decision tree file is from a YouTube tutorial.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('./Datasets/salaries.csv')

input_data = data.drop('salary_more_then_100k', axis='columns')
target_data = data['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

input_data['company'] = le_company.fit_transform(input_data['company'])
input_data['job'] = le_job.fit_transform(input_data['job'])
input_data['degree'] = le_degree.fit_transform(input_data['degree'])

X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size = 0.2, random_state = 42)

decision_tree = DecisionTreeClassifier()
model = decision_tree.fit(X_train, y_train)

prediction = model.predict(X_test)
print(prediction)