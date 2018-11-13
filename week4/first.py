import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

train = pandas.read_csv('salary-train.csv')
train.FullDescription = train.FullDescription.str.replace('[^a-zA-Z0-9]', ' ', regex=True)

vectorizer = TfidfVectorizer(min_df=5, lowercase=True)
X = vectorizer.fit_transform(train.FullDescription)

enc = DictVectorizer()
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_train = hstack([X, X_train_categ])

clf = Ridge(alpha=1)
clf.fit(X_train, train.SalaryNormalized)


test = pandas.read_csv('salary-test-mini.csv')
X_test_words = vectorizer.transform(test.FullDescription)
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_words, X_test_categ])

print('predicted', clf.predict(X_test))
