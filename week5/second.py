import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


data = pandas.read_csv('gbm-data.csv')
X = data.iloc[:, 1:].values
y = data.Activity.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

learning_rates = [1, 0.5, 0.3, 0.2, 0.1]


# draws overfitting graph when number of estimators increases
clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=0.2)
clf.fit(X_train, y_train)

pred_test = []
pred_train = []

min_val = 10000
min_i = 0
i = 0


for value in clf.staged_decision_function(X_test):
    current_val = log_loss(y_test, 1/(1 + np.exp(-value)))
    if current_val < min_val:
        min_val = current_val
        min_i = i
    pred_test.append(current_val)
    i+=1

# 36
print('min:', min_i, min_val)

for value in clf.staged_decision_function(X_train):
    pred_train.append(log_loss(y_train, 1/(1 + np.exp(-value))))

plt.figure()
plt.plot(pred_test, 'r', linewidth=2)
plt.plot(pred_train, 'g', linewidth=2)
plt.legend(['test', 'train'])
plt.show()

treeClf = RandomForestClassifier(n_estimators=36, random_state=241)
treeClf.fit(X_train, y_train)
predicted = treeClf.predict_proba(X_test)
print(log_loss(y_test, predicted))

