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
    return y_pred

# decision_classifier(train_X, train_y, test_X, test_y)

def naive_bayes(train_X, train_y, test_X, test_y):
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    from sklearn.metrics import classification_report, confusion_matrix
    model = MultinomialNB()
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    print("Classification report: \n", classification_report(test_y, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(test_y, y_pred))
    print("Score: \n", model.score(test_X, test_y))
    return y_pred

# naive_bayes(train_X, train_y, test_X, test_y)

def random_forest(train_X, train_y, test_X, test_y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    classifier_model = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=42)
    classifier_model.fit(train_X, train_y)
    y_pred = classifier_model.predict(test_X)
    print("Classification report: \n", classification_report(test_y, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(test_y, y_pred))
    print("Accuracy score: \n", accuracy_score(test_y, y_pred))
    return y_pred

y_pred = random_forest(train_X, train_y, test_X, test_y)

combined = pd.DataFrame(dict(actual=test_y, prediction=y_pred))

grouped_matches = df.groupby("team")
# group = grouped_matches.get_group("Manchester United")

def rolling_avgs(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed="left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
# rolling_avgs(group, cols, new_cols)

matches_rolling = df.groupby("team").apply(lambda x: rolling_avgs(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel("team")
matches_rolling.index = range(matches_rolling.shape[0])

def make_predictions(df, predictors):
    train = df[df["date"] < "2022-01-01"]
    test = df[df["date"] > "2022-01-01"]
    train_X = train[predictors]
    train_y = train["target"]
    test_X = test[predictors]
    test_y = test["target"]
    y_preds = random_forest(train_X, train_y, test_X, test_y)
    combined = pd.DataFrame(dict(actual=test_y, prediction=y_preds))
    return y_pred, combined

new_predictors = predictors + new_cols 
y_preds, combined = make_predictions(matches_rolling, new_predictors)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}

# this is done because there will be 2 rows that will display 1 match. i.e if Manchester United plays Arsenal, there will be one record in Arsenal group and one will be there in Manchester United group. For ex: Manchester United vs Arsenal at Old Trafford is same as Arsenal vs Manchester United (in arsenal group.)
# And there might be a possibility that the model predicts 2 different outcomes for the same match. To avoid that, we merge the same data.
mapping = MissingDict(**map_values)
combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
merged[(merged["prediction_x"] == 1) & (merged["prediction_y"] == 0)]["actual_x"].value_counts()