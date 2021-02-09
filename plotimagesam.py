from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

boston = load_boston();
X, y = boston.data, boston.target
n_samples, n_features = X.shape
print(n_samples, n_features)
print(boston.feature_names)
model = LinearRegression()
model.fit(X, y)
print(model.intercept_)
print(model.coef_.shape)
print(model.score(X, y))
iris = datasets.load_iris()
irisX = iris.data[:, :2] # make the data 2d
irisY = iris.target
irisX, irisY = irisX[irisY < 2], irisY[irisY < 2] # make it binary whatever that means
irisY[irisY == 0] = -1
print(irisX.shape)
logModel = LogisticRegression(C=1,)
logModel.fit(irisX, irisY)
intercept = logModel.intercept_
coefficient = logModel.coef_[0]
supportVectorMachine = SVC(kernel='linear', C=1, )
supportVectorMachine.fit(irisX, irisY)
svmIntercept = supportVectorMachine.intercept_
svmCoefficient = supportVectorMachine.coef_[0]
select = SelectPercentile(score_func = chi2, percentile = 16)
lr = LogisticRegression(tol = 1e-8, penalty = '12', C = 10., intercept_scaling=1_3)
char_vect = TfidfVectorizer(ngram_range=(1, 5), analyzer = 'char')
word_vect = TfidfVectorizer(ngram_range = (1, 3), analyzer = 'word', min_df = 3)
ft = FeatureUnion([('chars', char_vect), ('words', word_vect)])
clf = make_pipeline(ft, select, lr)
#scores = cross_val_score(clf, X, y, cv=2)
#print(np.mean(scores))
model = make_pipeline(PolynomialFeatures(degree=2), LogisticRegression())
X, y = make_moons(n_samples = 200, noise = 0.1, random_state = 0)
#plot_model(model, X, y)