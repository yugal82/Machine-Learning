# This regression file is from a YouTube tutorial.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

digits = load_digits()

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2, random_state = 42)

regressor = LogisticRegression()
model = regressor.fit(X_train, y_train)
y_pred = model.predict(X_test)
# print(model.score(X_test, y_test))

# plt.matshow(digits.images[421])
# plt.show()
# print(digits.target[421])

# print(model.predict(digits.data[[421]]))

cm = confusion_matrix(y_test, y_pred)
print(cm)