import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


data_set = pd.read_csv('Datasets/Bengaluru_House_Data.csv')

# Drop the columns which are not important such as 'availability', 'society', etc.
data_set = data_set.drop(columns=['availability','society', 'area_type', 'balcony'])

# data cleaning process is from here. The values which are null or NAN, we must handle such values by either dropping them or filling them with median or mean values.
# Here we have a big data set, so dropping 70-80 rows will not make any difference. Therefore we drop the values.
# print(data_set.isnull().sum())
data_set = data_set.dropna()

# now the 'size' column of the data set is a little different, it contains different values hence we need to handle that carefully.
print(data_set['size'].unique())

data_set['BHK'] = data_set['size'].apply(lambda x: x.split(' ')[0])
# print(data_set['BHK'])

data_set = data_set.drop(columns=['size'])


# in 'total_sqft' column, we have some values that are given in range, for example 1134 - 1203, which will cause problems to build the model. Hence, we need to convert them to numbers and we do that by taking the average of the values.
# To check which values are given in range, we define a function that tries to convert the given input to float. If the input is not a range then it will return true otherwise it will return false. 
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

print(data_set[~data_set['total_sqft'].apply(is_float)])
# output -> [190 rows x 6 columns]
# so there are 190 rows where the 'total_sqft' is given as a range.

# Now to take the average of the range and convert to single value, we write a function that calculates the average.
def calculateAvg(x):
    values = x.split('-')
    if len(values) == 2:
        return (float(values[0]) + float(values[1]))/2
    try:
        return float(x)
    except:
        return None 

data_set['total_sqft'] = data_set['total_sqft'].apply(calculateAvg)
# print(data_set['total_sqft'])

# ------------------------------------------------------------------------------------------
# AT THIS STAGE, WE HAVE HANDLED WITH THE NULL VALUES, CONVERTED THE 'size' COLUMN, AND CONVERTED RANGES TO A FLOAT VALUE IN 'total_sqft' COLUMN
#-------------------------------------------------------------------------------------------

data_set_2 = data_set.copy()
data_set_2['price_per_sqft'] = (data_set['price']/data_set['total_sqft'])*100000

print(data_set_2['location'].unique())
# The dataset contains 1300+ different values for 'location' column. As these are categorical variables, we can encode them using one hot encoding. However, 1300+ values means it will generate 1300+ new columns to achieve that.
# To avoid this, we decide a threshold value, and all values for 'location' column that are less than this threshold will be put into 'others' column.
# For eg. location value that comes more than 50 are not put into 'others' column.

data_set_3 = data_set_2.copy()
data_set_3['location'] = data_set_3['location'].apply(lambda x: x.strip())
location_stats = data_set_3.groupby('location')['location'].agg('count').sort_values(ascending=False)
# print(location_stats)
# location
# Whitefield               535
# Sarjapur  Road           392
# Electronic City          304
# Kanakpura Road           266
# Thanisandra              236
#                         ...
# 1 Giri Nagar               1
# Kanakapura Road,           1
# Kanakapura main  Road      1
# Karnataka Shabarimala      1
# whitefiled                 1
# Name: location, Length: 1293, dtype: int64

# So here 'Whitefield' value is present in 535 rows and '1 Giri Nagar' value is present in only 1 row.
# The values that are below the threshold are moved to 'other' column.
print(len(location_stats[location_stats<=10]))
# output -> 1052
# There are 1052 values that have 'location' value less than 10.

location_stats_less_than_10 = location_stats[location_stats<=10]

data_set_3['location'] = data_set_3['location'].apply(lambda x: 'Other' if x in location_stats_less_than_10 else x)
# print(len(data_set_3['location'].unique()))


# -----------------------------------------------------------------------------
# UPTIL HERE, WE PREFORMED DATA PREPROCESSING AND DIMESIONALITY REDUCTION.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# NOW LETS DETECT THE OUTLIERS AND REMOVE THEM
# -----------------------------------------------------------------------------

data_set_3['BHK'] = data_set_3['BHK'].astype(int)

# print(data_set_3[data_set_3['total_sqft']/data_set_3['BHK'] < 300].head())
# output ->                location  total_sqft  bath  price  BHK  price_per_sqft
        # 9                 Other      1020.0   6.0  370.0  6.0    36274.509804
        # 45           HSR Layout       600.0   9.0  200.0  8.0    33333.333333
        # 58        Murugeshpalya      1407.0   4.0  150.0  6.0    10660.980810
        # 68  Devarachikkanahalli      1350.0   7.0   85.0  8.0     6296.296296
        # 70                Other       500.0   3.0  100.0  3.0    20000.000000

# As you can see above, there are 5 values with bedrooms less than 300 sq.ft which is considered as outliers. Since generally each bedroom is contains around 300sq.ft area. So these are outliers and we must remove them.
data_set_3 = data_set_3[~(data_set_3['total_sqft']/data_set_3['BHK'] < 300)]

# The values where price_per_sqft is extremely low or extremely high, these values are also considered to be outliers in the data set.
# print(data_set_3['price_per_sqft'].describe())
# count     12456.000000
# mean       6308.502826
# std        4168.127339
# min         267.829813
# 25%        4210.526316
# 50%        5294.117647
# 75%        6916.666667
# max      176470.588235

# now here the min is 267 rupees per sq ft which is very less likely to be true in Bengaluru.
# similarly with max value - 1,76,470.
# generllay if the dataset is NORMALLY DISTRIBUTED, the values that are outside mean + 1SD range are considered as outliers. (SD -> Standard deviation)
# so to tackle this problem, we write a function to remove the outliers

def remove_per_sq_ft_outliers(data):
    df_out = pd.DataFrame()
    for key, subdf in data.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - sd)) & (subdf.price_per_sqft <= (m + sd))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df4 = remove_per_sq_ft_outliers(data_set_3)
# print(df4.shape)

# Now let's remove outliers where the number of bathroom in the house are 2 more than the number of bedrooms.
# For eg. If you have 10 bedrooms and 12 bathrooms, we consider it as outlier.
# i.e bathroom - bedroom > 2, mark it as outlier.

# print(df4.bath.unique())
# print(df4[df4.bath > 10])
# print(df4[df4.bath > df4.BHK+2])
        #    location  total_sqft  bath   price  BHK  price_per_sqft
# 1626  Chikkabanavar      2460.0   7.0    80.0  4.0     3252.032520
# 5238     Nagasandra      7000.0   8.0   450.0  4.0     6428.571429
# 5850          Other     11338.0   9.0  1000.0  6.0     8819.897689

df4 = df4[df4.bath < df4.BHK+2]


# Now the 'location' column is in string format, but ML model expects data to be numerical. So we need to convert it to numerical values. As it is a categorical column, we will use One Hot Encoding.

dummies = pd.get_dummies(df4['location'], dtype=int)
# we created a new data frame for dummies and we will append this data frame to the original data frame 'df5'

df5 = pd.concat([df4, dummies.drop('Other', axis='columns')], axis='columns')
df5 = df5.drop('location', axis='columns')

df5 = df5.drop('price_per_sqft', axis='columns')

# X and y are independent and dependent variables respectively.
X = df5.drop(['price'], axis='columns')
y = df5.price

# print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# print(cross_val_score(model, X, y, cv=cv))


def find_best_model(X, y):
    algo = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algo.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

# print(find_best_model(X,y))
# The GridSearchCV suggests that the best technique is LinearRegression() model.

# so we use the above trained model to predict the values for test data

def predict_price(location,total_sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# print(predict_price('1st Phase JP Nagar', 1000, 2, 2))
# print(predict_price('1st Phase JP Nagar', 1000, 3, 3))
# print(predict_price('Indira Nagar', 1000, 2, 2))
# print(predict_price('Indira Nagar', 1000, 3, 3))

# HERE THE MODEL BUILDING PROCESS IS DONE AND WE EXPORT THE MODEL IN A PICKLE FILE.
import pickle
with open('Projects\Project-1\Bengaluru_house_price_predict.pickle', 'wb') as f:
    pickle.dump(model, f)

import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open('Projects\Project-1\columns.json', 'w') as f:
    f.write(json.dumps(columns))
