import pandas
from sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv', header=None)

X = data.iloc[:, 1:]
y = data.iloc[:,0]

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, y)
print(X)
print(clf.support_vectors_)  # 4 5 10 (+1 because starts from 0)
