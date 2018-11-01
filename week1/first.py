import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# How many men and women were on the ship?
print(data['Sex'].value_counts())

# How many passengers survived in percent?
total_amount = data['Survived'].size
survived = data['Survived'].value_counts().tolist()[1]
print('survived:', survived / total_amount * 100)

# How many first class passengers were there in percent?
first_class = dict(data['Pclass'].value_counts())
print('first class:', first_class[1] / total_amount * 100)

# Calculate mean and median of passengers' age.
print('age mean:', data['Age'].mean())
print('age median:', data['Age'].median())

# Discover if there's a correlation between amount of siblings/spouses and
print('correlation:', data['Parch'].corr(data['SibSp']))

# Find most popular female name.
females = data.loc[data['Sex'] == 'female']
names_list = females['Name'].value_counts().keys().tolist()
import re
names_dict = {}
for name in names_list:
    first_names = re.split('\W+',name.split('.')[1].strip())
    for first_name in first_names:
        if (len(first_name) != 0):
            add = names_dict.get(first_name) if names_dict.get(first_name) else 0
            names_dict[first_name] = add + 1
max = 0
max_name = ''
for name in names_dict:
    if (max < names_dict[name]):
        max = names_dict[name]
        max_name = name
print('most common female name:', max_name)

