import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('./Datasets/titanic.csv')
# print(data.head())

# drop the columns which are not required
data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Embarked', 'Ticket', 'Cabin'], axis='columns',inplace=True)

# separate out target variables and input features
target = data['Survived']
input_data = data.drop('Survived', axis='columns')

# print("Target: \n", target)
# print("input_data: \n", input_data)

le = LabelEncoder()
input_data['Sex'] = le.fit_transform(input_data['Sex'])

# Fill the missing and NaN values
# print(input_data.isnull().sum())
input_data['Age'] = input_data['Age'].fillna(value=input_data['Age'].mean())
input_data['Fare'] = input_data['Fare'].fillna(value=input_data['Fare'].mean())
# print(input_data.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(input_data, target, test_size=0.2)

print(len(X_train))
print(len(X_test))

model = GaussianNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(y_test[:10])
print(y_pred[:10])