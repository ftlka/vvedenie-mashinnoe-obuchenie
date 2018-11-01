# We need to choose two params which correlate the most with Survived field
from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas.read_csv('titanic.csv')
data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna()
genders = {'female': 0, 'male': 1}

# to represent gender as int
data = data.applymap(lambda s: genders.get(s) if s in genders else s)
df = data[['Pclass', 'Fare', 'Age', 'Sex']]
y = data['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf = clf.fit(df, y)
print(clf.feature_importances_)

