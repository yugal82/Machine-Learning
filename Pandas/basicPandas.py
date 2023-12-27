import pandas as pd

# df = pd.read_csv('./Datasets/IRIS.csv')
# print(df.head())

weather = {
    'day': ['12/12/2021', '13/12/2021', '14/12/2021', '15/12/2021', '16/12/2021'],
    'temperature': [31,35,29,30,32],
    'windspeed': [7,6,3,8,8],
    'event': ['Rain', 'Sunny', 'Snow', 'Rain', 'Snow']
}

df2 = pd.DataFrame(weather)
print(df2.head())
print(df2.shape)
print(df2.columns)
print(df2.dtypes)

print(type(df2.event)) # output: <class 'pandas.core.series.Series'> ... The columns in the dataframe are of type pandas series.

print(df2.describe())
print(df2['windspeed'].max())
print(df2['windspeed'].min())
print(df2['windspeed'].mean())
print(df2['windspeed'].std())

# conditionally printing the data is kind of similar to SQL query statements.
print(df2[df2['windspeed']>6])
#           day  temperature  windspeed event
# 0  12/12/2021           31          7  Rain
# 3  15/12/2021           30          8  Rain
# 4  16/12/2021           32          8  Snow

# Creating a pandas dataframe from a python tuple
weather_data = [
    ('12/12/2021', 28, 5, 'Rain'),
    ('13/12/2021', 31, 7, 'Snow'),
    ('14/12/2021', 32, 8, 'Sunny'),
]

df3 = pd.DataFrame(weather_data, columns=['day', 'temperature', 'windspeed', 'event'])
# while creatint a dataframe froma python tuple, we need to specify the columns in the pd.DataFrame() method.
print(df3)


print('---------------------- New Functions ----------------------------------------------------------------')
# group by functionality
df4 = pd.read_csv('.\Pandas\weather_Data.csv')
grouped = df4.groupby(['city'])
for city, city_df in grouped:
    print(city)
    print(city_df)

# print(grouped.max())
# print(grouped.describe())