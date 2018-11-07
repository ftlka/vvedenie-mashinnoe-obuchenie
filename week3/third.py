import pandas
import math
from sklearn.metrics import roc_auc_score

data = pandas.read_csv('data-logistic.csv', header=None)

x1 = data.values[:, 1:2]
x2 = data.values[:, 2:]
y = data.values[:,0]


def calculate_w_next(w, k, w1, w2, x, c):
    l = len(y)
    sum = 0
    for i in range(l):
        sum += y[i] * x[i][0] * (1 - 1 / (1 + math.exp(-y[i] * (w1 * x1[i][0] + w2 * x2[i][0]))))
    return w + (k / l) * sum - k * c * w


def calc_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def log_regression(k, w1, w2, epsilon, c, l):
    i = 1

    while (i <= l):
        w1_next = calculate_w_next(w1, k, w1, w2, x1, c)
        w2_next = calculate_w_next(w2, k, w1, w2, x2, c)
        if (calc_dist(w1, w2, w1_next, w2_next) < epsilon):
            break
        w1 = w1_next
        w2 = w2_next
        i+=1

    print(w1, w2)
    predicted_x = []
    for i in range(0, len(y)):
        pred_x = 1 / (1 + math.exp(-w1 * x1[i][0] - w2 * x2[i][0]))
        predicted_x.append(pred_x)

    return predicted_x


standart_c = log_regression(0.1, 0, 0, 0.00001, 0, 10000)
print(roc_auc_score(y, standart_c))
increased_c = log_regression(0.1, 0, 0, 0.00001, 10, 10000)
print(roc_auc_score(y, increased_c))

