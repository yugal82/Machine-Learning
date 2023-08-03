# This regression file is from a YouTube tutorial.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('./Datasets/canada_per_capita_income.csv')
# print(data.head(5))

regressor = LinearRegression()
regressor.fit(data[['year']], data[['per capita income (US$)']])

plt.scatter(x=data['year'], y=data['per capita income (US$)'])
plt.xlabel('Year')
plt.ylabel('Per Capita Income (USD)')
plt.title('Year vs Per Capita Income')
plt.plot(data[['year']], regressor.predict(data[['year']]), color='red')
plt.show()


print(regressor.predict([[2020]]))