from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#scatter_matrix(dataset)
#pyplot.show()

dataset['height-weight'] = dataset['height'] / dataset['weight']
features = ['age', 'gender', 'height-weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
model = RandomForestClassifier(n_estimators=100)
X_train = dataset[features]
Y_train = dataset['cardio']
model.fit(X_train, Y_train)
test['height-weight'] = test['height'] / test['weight']
test_train = test[features]
predictions = model.predict(test_train)
ids = []
var = 0
for row in test.iterrows():
    ids.append(var)
    var = var + 1

output = pd.DataFrame({'id' : ids, 'cardio' : predictions})
output.to_csv('predictions.csv', index = False)

print(dataset.head())

print('correlation matrix displayed')