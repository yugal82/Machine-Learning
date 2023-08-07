import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv('./Datasets/IRIS.csv')

le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])

df0 = data[data.species == 0]
df1 = data[data.species == 1]
df2 = data[data.species == 2]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal_length'], df0['sepal_width'], color = 'green')
plt.scatter(df1['sepal_length'], df1['sepal_width'], color = 'blue')
# plt.show()

plt.xlabel('Pepal Length')
plt.ylabel('Pepal Width')
plt.scatter(df0['petal_length'], df0['petal_width'], color = 'green')
plt.scatter(df1['petal_length'], df1['petal_width'], color = 'blue')
# plt.show()

data_y = data['species']
data_x = data.drop('species', axis = 'columns')

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

classifier = SVC()
model = classifier.fit(X_train, y_train)
print(model.predict(X_test))