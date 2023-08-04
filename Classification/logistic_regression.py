# This regression file is from a YouTube tutorial.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('./Datasets/insurance_data.csv')

# plt.scatter(x=data['age'], y=data['bought_insurance'])
# plt.xlabel('Age')
# plt.ylabel('Insurance bought (Y or N)')
# plt.title('Age vs Insurance bought')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(data[['age']], data[['bought_insurance']], test_size = 0.2, random_state = 42)

regressor = LogisticRegression()
model = regressor.fit(X_train, y_train)
print(model.score(X_test, y_test))

prediction = model.predict(X_test)
print(prediction)
print(model.predict_proba(X_test))