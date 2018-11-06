import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)
X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# getting optimal C
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# gs.fit(X, y)  # this function takes 6-7 minutes to compute
# print(gs.cv_results_) # last 6 values have best score of 0.99328108, we take min: c = 1
# print('best score:', gs.best_score_)
clf = SVC(C=1, kernel='linear', random_state=241)
clf.fit(X, y)

top10 = np.argsort(abs(clf.coef_.toarray()[0]))[-10:]
feature_mapping = vectorizer.get_feature_names()
for i in top10:
    print(feature_mapping[i])


