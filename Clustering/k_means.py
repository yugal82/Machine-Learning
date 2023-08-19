import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('./Datasets/income.csv')
# print(data.head(5))

# plt.scatter(data['Age'], data['Income($)'])
# plt.show()

# data preprocessing, normalizing data
scaler = MinMaxScaler()

scaler.fit(data[['Income($)']])
data['Income($)'] = scaler.transform(data[['Income($)']])

scaler.fit(data[['Age']])
data['Age'] = scaler.transform(data[['Age']])
# print(data.head())


# performing k_means clustering
k_means = KMeans(n_clusters = 3)
y_pred = k_means.fit_predict(data[['Age', 'Income($)']])
print(y_pred)

centroids = k_means.cluster_centers_
print(centroids)

data['cluster'] = y_pred
print(data)
df1 = data[data.cluster == 0]
df2 = data[data.cluster == 1]
df3 = data[data.cluster == 2]

plt.scatter(df1['Age'], df1['Income($)'], color='blue')
plt.scatter(df2['Age'], df2['Income($)'], color='green')
plt.scatter(df3['Age'], df3['Income($)'], color='red')
plt.show()

k_range = range(1,10)
sse = []  #sse ---> Square Sum Error

for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(data[['Age', 'Income($)']])
    sse.append(km.inertia_)

plt.plot(k_range, sse)
plt.xlabel('K')
plt.ylabel('SSE')
plt.show()