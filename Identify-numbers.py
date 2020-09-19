import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn

from sklearn import preprocessing, metrics, cross_validation
from sklearn.datasets import fetch_mldata
from scipy.special import comb
from sklearn.cross_validation import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import SGD

mnist = fetch_mldata('MNIST original')

n = len(mnist.data)
N = 10000
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

n_in = len(X[0])
n_hidden = 200
n_out = len(Y[0])

model = Sequential()
model .add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('tanh'))

model .add(Dense(n_hidden))
model.add(Activation('tanh'))

model .add(Dense(n_hidden))
model.add(Activation('tanh'))

model .add(Dense(n_hidden))
model.add(Activation('tanh'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])


epochs = 1000
batch_size = 100

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)


loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)