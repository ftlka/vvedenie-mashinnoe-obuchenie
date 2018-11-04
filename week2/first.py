import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from numpy import mean
from sklearn.preprocessing import scale

data = pandas.read_csv('wine.data')

Y = data.iloc[:,0]
X = data.iloc[:, 1:]
# neigh.fit(X, y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i in range(1, 51):
    # we are searching for max quality with different amount of neighbours
    neigh = KNeighborsClassifier(n_neighbors=i)
    # creates set of splits
    res_arr = cross_val_score(neigh, X, Y, cv=kf, scoring='accuracy')
    print(i, mean(res_arr)) # i = 1 - max 0.7352380952380952

# normalize
X = scale(X)

max_i = 0
max_mean = 0
for i in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=i)
    res_arr = cross_val_score(neigh, X, Y, cv=kf, scoring='accuracy')
    if (max_mean < mean(res_arr)):
        max_i = i
        max_mean = mean(res_arr)
    print(i, mean(res_arr))

print(max_i, max_mean)