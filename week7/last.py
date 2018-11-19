import pandas
import datetime
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


df = pandas.read_csv('./features.csv', index_col='match_id')

# deleting columns with match results
X = df.drop(columns=[
    'duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
    'barracks_status_radiant', 'barracks_status_dire'
])

# getting all features with blanks
features_with_blanks = []
for idx, column in enumerate(X.count()):
    if column != 97230:
        features_with_blanks.append(X.columns[idx])

# filling up blanks with 0s
X = X.fillna(0)

# assigning y vector
y = df.radiant_win

# creating 5-fold partitioning with shuffle
kf = KFold(n_splits=5, shuffle=True)

# splitting data into train and test
X_arr = np.asarray(X)
y_arr = np.asarray(y)
for train_index, test_index in kf.split(X_arr):
    X_train, X_test = X_arr[train_index], X_arr[test_index]
    y_train, y_test = y_arr[train_index], y_arr[test_index]

# getting first half of training data to make classification faster
# X_train = X_train[len(X_train)//2:]
# y_train = y_train[len(y_train)//2:]

# checking quality for gradient boosting classifier with 10, 20 and 30 trees
# and setting up timers
tree_sizes = [10, 20, 30]
for tree_size in tree_sizes:
    start_time = datetime.datetime.now()

    #changed max_depth for faster performance
    clf = GradientBoostingClassifier(n_estimators=tree_size, max_depth=2)
    clf.fit(X_train, y_train)

    pred_y = clf.predict_proba(X_test)[:, 1]

    print(roc_auc_score(y_true=y_test, y_score=pred_y))
    print('Time for', tree_size, 'number of trees', datetime.datetime.now() - start_time)


# Logistic regression part
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# normalizing X and splitting into test and train
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_arr = np.asarray(X)
y_arr = np.asarray(y)
for train_index, test_index in kf.split(X_arr):
    X_train, X_test = X_arr[train_index], X_arr[test_index]
    y_train, y_test = y_arr[train_index], y_arr[test_index]


def findBestC(X_train, X_test, y_train, y_test):
    # getting best c value - it will be around 0.01 - 1
    c_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    answer = []

    max_c = -1
    max_quality = 0

    for value in c_values:
        start_time = datetime.datetime.now()

        # specified a solver because of warnings in console
        logistic_clf = LogisticRegression(penalty='l2', C=value, solver='lbfgs')
        logistic_clf.fit(X_train, y_train)

        pred_y = logistic_clf.predict_proba(X_test)[:, 1]
        quality = roc_auc_score(y_true=y_test, y_score=pred_y)

        if quality > max_quality:
            max_quality = quality
            max_c = value
            answer = pred_y

        # print('Time for c =', value, datetime.datetime.now() - start_time)

    print(max_c, max_quality)
    return answer

findBestC(X_train, X_test, y_train, y_test)


# deleting categorical features
X = df.drop(columns=[
    'duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
    'barracks_status_radiant', 'barracks_status_dire', 'lobby_type',
    'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
    'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'
])

X = X.fillna(0)
scaler.fit(X)
X = scaler.transform(X)

X_arr = np.asarray(X)
y_arr = np.asarray(y)
for train_index, test_index in kf.split(X_arr):
    X_train, X_test = X_arr[train_index], X_arr[test_index]
    y_train, y_test = y_arr[train_index], y_arr[test_index]

findBestC(X_train, X_test, y_train, y_test)


# calculating amount of heroes
heroes = [
    'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
    'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'
]

unique_heroes = []
for hero in heroes:
    current = np.asarray(df[hero])
    unique_heroes = unique_heroes + list(set(current) - set(unique_heroes))

N = unique_heroes[-1]

# bag of words implementation
X_pick = np.zeros((df.shape[0], N))

for i, match_id in enumerate(df.index):
    for p in range(5):
        X_pick[i, df.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, df.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1


# adding bag of words to X
X = np.hstack((X, X_pick))

scaler.fit(X)
X = scaler.transform(X)

X_arr = np.asarray(X)
y_arr = np.asarray(y)
for train_index, test_index in kf.split(X_arr):
    X_train, X_test = X_arr[train_index], X_arr[test_index]
    y_train, y_test = y_arr[train_index], y_arr[test_index]

pred_y = findBestC(X_train, X_test, y_train, y_test)

# calculating min and max probability
min = 2
max = 0
for e in pred_y:
    if e > max:
        max = e
    elif e < min:
        min = e

print(max, min)
