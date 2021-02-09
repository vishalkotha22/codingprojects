# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import SVC

train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('testing.csv')

y_train = train_data['Survived']

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
train_data['Embarked'] = pd.get_dummies(train_data['Embarked'].fillna(train_data['Embarked'].mode()))
test_data['Embarked'] = pd.get_dummies(test_data['Embarked'].fillna(test_data['Embarked'].mode()))
train_data['Sex'] = pd.get_dummies(train_data['Sex'])
test_data['Sex'] = pd.get_dummies(test_data['Sex'])
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Fare']
x_train = train_data[features]
x_test = test_data[features]

model = SVC(kernel='rbf')
model.fit(x_train, y_train)
results = model.predict(x_test)

output = pd.DataFrame({'PassengerId' : test_data.PassengerId, 'Survived' : results.tolist()})
output.to_csv("liltjay.csv", index = False)