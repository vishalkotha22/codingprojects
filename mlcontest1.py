from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

dataset = pd.read_csv('train.csv')

scatter_matrix(dataset)
pyplot.show()

print('correlation matrix displayed')