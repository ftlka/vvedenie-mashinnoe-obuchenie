import pandas
from sklearn.decomposition import PCA
from numpy import corrcoef
import numpy

train = pandas.read_csv('close_prices.csv').iloc[:, 1:]

cls = PCA(n_components=10)
cls.fit(train)
train = cls.transform(train)

first_comp = []
for ar in train:
    first_comp.append(ar[0])

jones = pandas.read_csv('djia_index.csv').iloc[:, 1:]

jones_arr = []
for ar in numpy.array(jones):
    jones_arr.append(ar[0])

print('first coef', corrcoef(jones_arr, first_comp))

max_val = -1000
max_idx = -1
for idx in range(0, len(cls.components_[0])):
    val = cls.components_[0][idx]
    if (val > max_val):
        max_val = val
        max_idx = idx

print(max_idx)
