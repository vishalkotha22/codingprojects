import pandas as pd
import numpy as np

train_data = pd.read_csv('train.csv')
temp_train = list(list())
y_train = list(list())
for row in train_data.iterrows():
    temp = [0] * 10
    row = row[1]
    temp[row[0]] = 1
    y_train.append(temp)
    pixels = list()
    for pixel in row[1:]:
        pixels.append(float(pixel)/255.0)
    temp_train.append(pixels)

x_train = np.array(temp_train)

test_data = pd.read_csv('test.csv')
temp_test = list(list())
for row in test_data.iterrows():
    row = row[1]
    pixels = list()
    for pixel in row[1:]:
        pixels.append(float(pixel)/255.0)
    temp_test.append(pixels)

x_test = np.array(temp_test)

print('data processed')

def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoid_derv(s):
    return s * (1 - s)


def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res / n_samples


def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss


class MyNN:
    def __init__(self, x, y):
        self.x = x
        self.lr = 0.5

        self.w1 = np.random.randn(784, 128)
        self.b1 = np.zeros((1, 128))
        self.w2 = np.random.randn(128, 128)
        self.b2 = np.zeros((1, 128))
        self.w3 = np.random.randn(128, 10)
        self.b3 = np.zeros((1, 10))
        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

    def backprop(self):
        loss = error(self.a3, self.y)
        print('Error :', loss)
        a3_delta = cross_entropy(self.a3, self.y)  # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2)  # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1)  # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()


model = MyNN(x_train, np.array(y_train))

epochs = 2000
for x in range(epochs):
    print(x)
    model.feedforward()
    model.backprop()

predictions = list()
for test in x_test:
    print(model.predict(test))
    predictions.append(model.predict(test))

output = pd.DataFrame({'id' : test_data.id, 'number' : predictions})
output.to_csv('gunnawunna.csv', index = False)
print('done')