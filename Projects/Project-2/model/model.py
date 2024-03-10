import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/PL_matches.csv', index_col=0)
# print(df.head())
# print(df.shape)  --> (1389, 27)

# each team plays 38 matches
# 38*20*2 = 1520 (multiplied by 2 because the data is for 2 seasons)
# but there are only 1389 matches, so we have missing data. To check missing data, we see how many matches each team have played in the dataset

df['team'].value_counts()
# team
# Southampton                 72
# Brighton and Hove Albion    72
# Manchester United           72
# West Ham United             72
# Newcastle United            72
# Burnley                     71
# Leeds United                71
# Crystal Palace              71
# Manchester City             71
# Wolverhampton Wanderers     71
# Tottenham Hotspur           71
# Arsenal                     71
# Leicester City              70
# Chelsea                     70
# Aston Villa                 70
# Everton                     70
# Liverpool                   38
# Fulham                      38
# West Bromwich Albion        38
# Sheffield United            38
# Brentford                   34
# Watford                     33
# Norwich City                33
# Name: count, dtype: int64

df[df['team'] == "Liverpool"]

df["date"] = pd.to_datetime(df['date'])
# print(df.dtypes)

df['venue'] = df['venue'].astype('category').cat.codes
df['opp_code'] = df['opponent'].astype('category').cat.codes
df["time"] = df["time"].str.replace(":.+", "", regex=True).astype("int")

# df["date"].dt.dayofweek
df["day_code"] = df["date"].dt.dayofweek

df["target"] = df["result"].replace({"L": 0, "D": 0, "W": 2})

# here we have to be careful while splitting the data into train and test data. As train_test_split randomly splits the data into train and test, this can be inaccurate as we have to train the model on past data and then predict on future data. It cannont be train on future data and then predict on past data (which is illogical)
# so we spilt the train data that is before "2022-01-01" and test data that is after "2022-01-01", i.e train data is full of past data and test data is full of future data.
train = df[df["date"] < "2022-01-01"]
test = df[df["date"] > "2022-01-01"]

predictors = ["venue", "opp_code", "time", "day_code"]

train_X = train[predictors]
train_y = train[["target"]]
test_X = test[predictors]
test_y = test[["target"]]

def decision_classifier(train_X, train_y, test_X, test_y):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    classifier = DecisionTreeClassifier(max_depth=3, min_samples_split=5)
    classifier.fit(train_X, train_y)
    y_pred = classifier.predict(test_X)
    print("Classification report: \n", classification_report(test_y, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(test_y, y_pred))
    print("Score: \n", classifier.score(test_X, test_y))

# decision_classifier(train_X, train_y, test_X, test_y)

from sklearn.naive_bayes import MultinomialNB, GaussianNB
def naive_bayes(train_X, train_y, test_X, test_y):
    from sklearn.metrics import classification_report, confusion_matrix
    model = MultinomialNB()
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    print("Classification report: \n", classification_report(test_y, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(test_y, y_pred))
    print("Score: \n", model.score(test_X, test_y))

naive_bayes(train_X, train_y, test_X, test_y)

def random_forest(train_X, train_y, test_X, test_y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    classifier_model = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=42)
    classifier_model.fit(train_X, train_y)
    y_pred = classifier_model.predict(test_X)
    print("Classification report: \n", classification_report(test_y, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(test_y, y_pred))
    print("Accuracy score: \n", accuracy_score(test_y, y_pred))

# random_forest(train_X, train_y, test_X, test_y)