from sklearn.tree import DecisionTreeClassifier
from sklearn import EstimatorObject

dtree = DecisionTreeClassifier()
dtree.fit(df(['Age', 'Distance']), df['Attended'])

model = EstimatorObject()


cart_plot(dtree)