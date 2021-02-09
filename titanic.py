import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans
df = pd.read_csv('train.csv')
columns_target = ['Survived']
columns_train = ['Age', 'Pclass', 'Sex', 'Fare']
X = df[columns_train]
Y = df[columns_target]
X['Age'] = X['Age'].fillna(X['Age'].median())
d = {'male' : 0, 'female' : 1}
X['Sex'] = X['Sex'].apply(lambda x : d[x])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.47, random_state = 42)
model = svm.LinearSVC()
model.fit(X_train, Y_train)
passId = []
i = 1
while i <= 418:
    passId.append(i)
    passId.append("\n")
    i += 1
data = pd.DataFrame({'Survived' :  model.predict(X_test)})
predictions = []
for row in data.iterrows():
    predictions.append(row)
del predictions[418]
datafra = pd.DataFrame({'Survived' : predictions})
datafra.index += 1
datafra.to_csv('answer.csv', index=True, index_label = 'PassengerId')
print(passId)

stuff = df.values
print(stuff)
kmeans = KMeans() #8 clusters is default
kmeansCluster = KMeans(n_clusters=11)
kmeans.fit(stuff)
# silhoutte score is used for test of fit, like cross val scores is used for fit teests
#Use Lingo3D for hierarchial clustering