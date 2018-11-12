import pandas
from sklearn import metrics

data = pandas.read_csv('classification.csv')

true = data.iloc[:, 0]
pred = data.iloc[:, 1]

true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0

for idx in range(0, len(true)):
    current_true = true[idx]
    current_pred = pred[idx]
    if (current_true == 1):
        if (current_pred == 1):
            true_positives += 1
        else:
            false_negatives += 1
    else:
        if (current_pred == 0):
            true_negatives += 1
        else:
            false_positives += 1
print('TP:', true_positives, "FP:", false_positives, "FN:", false_negatives, "TN:", true_negatives)


accuracy = metrics.accuracy_score(true, pred)
precision = metrics.precision_score(true, pred)
recall = metrics.recall_score(true, pred)
f_measure = metrics.f1_score(true, pred)
print(accuracy, precision, recall, f_measure)


data = pandas.read_csv('scores.csv')

true = data.true
score_logreg = data.score_logreg
score_svm = data.score_svm
score_knn = data.score_knn
score_tree = data.score_tree

print('logreg', metrics.roc_auc_score(true, score_logreg))
print('svm', metrics.roc_auc_score(true, score_svm))
print('knn', metrics.roc_auc_score(true, score_knn))
print('tree', metrics.roc_auc_score(true, score_tree))


def find_max_precision(score):
    [precision, recall, threshold] = metrics.precision_recall_curve(true, score)
    max_precision = 0
    for idx in range(0, len(precision)):
        if (recall[idx] >= 0.7 and precision[idx] > max_precision):
            max_precision = precision[idx]
    return max_precision


print('logreg max precision', find_max_precision(score_logreg))
print('svm max precision', find_max_precision(score_svm))
print('knn max precision', find_max_precision(score_knn))
print('tree max precision', find_max_precision(score_tree))



