# Read data

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('rrMapBlank.PNG')
plt.imshow(img)
plt.show()

import csv
import tensorflow
import numpy as np

rows = []
with open('training.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        rows.append(row)

print('read data')

X = []
y = []
for i in range(len(rows)):
    if i == 0:  # Skip header!
        continue
    image = row[30].split(" ")
    X.append(np.reshape(image, (96, 96)))
    y.append(row[0:30])

print('spliced data')

from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
model = Sequential()

model.add(Convolution2D(32, (3,3), activation = 'relu', padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3,3), activation = 'relu', padding='same', use_bias=False))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), activation = 'relu', padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3), activation = 'relu', padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

print('created model')

#Train model
model.fit(X, y, epochs = 10, batch_size = 32, validation_split = 0.2)

print('trained model')

test_data = []
with open('testing.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimeter=',')
    for row in reader:
        test_data.append(row)

print('read test data')

y_test = []
for row in test_data:
    y_test.append(model.predict(row))

print('predicted test data')

output = []
header = []
header.append("ImageID.FeatureID")
header.append("Value")
output.append(header)
for i in range(2049):
    for x in range(30):
        row = []
        row.append(str(i + 1) + "." + str(x + 1))  # We start Ids at 1, so we need to add 1 to each value
        row.append(y_test[i][x])
        output.append(row)

print('converted test data to submission format')

with open('wassupp.csv', 'w', newline='\n') as f:
    writer = csv.writer(f)
    writer.writerows(output)

print('submission file created')