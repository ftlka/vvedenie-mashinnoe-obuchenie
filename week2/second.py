from sklearn import datasets
from sklearn.preprocessing import scale
from numpy import linspace
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean

df = datasets.load_boston()
X = df.data
y = df.target

# normalizing
X = scale(X)

# get results of p from 1 to 10 with 200 total values
p_values = linspace(1, 10, 200)

max_p = 0
max_val = -1000
for p in p_values:
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    res = cross_val_score(regressor, X, y, cv=kf, scoring='neg_mean_squared_error')
    print(p, mean(res))
    if (mean(res) > max_val):
        max_val = mean(res)
        max_p = p

print(max_p, max_val)
