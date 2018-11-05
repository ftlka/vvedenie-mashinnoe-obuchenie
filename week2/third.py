import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def compute_accuracy(X_train, y_train, X_test, y_test):
    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    # getting result by perceptron
    predictions = clf.predict(X_test)
    # how well we predicted
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

train = pandas.read_csv('perceptron-train.csv', header=None)
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]

test = pandas.read_csv('perceptron-test.csv', header=None)
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]


original_accuracy = compute_accuracy(X_train, y_train, X_test, y_test)
print('original accuracy:', original_accuracy)

# now we will sclale X and see how accuracy improved
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

accuracy_scaled = compute_accuracy(X_train_scaled, y_train, X_test_scaled, y_test)
print('scaled:', accuracy_scaled)

print('difference:', accuracy_scaled - original_accuracy)
