import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data.iloc[:, :-1]
y = data.Rings

kf = KFold(n_splits=5, random_state=1, shuffle=True)

for num_trees in range(1, 51):
    clf = RandomForestRegressor(n_estimators=num_trees, random_state=1)
    res = np.mean(cross_val_score(clf, X, y, cv=kf, scoring='r2'))
    if res > 0.52:
        print(num_trees)
        break


# For some reason tests can be passed with answer + 1
