import keras
import numpy as np
import tensorflow
import tflearn
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizer_v1 import SGD


def create_model(training, output):
    training = np.array(training)
    output = np.array(output)

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 20)
    net = tflearn.fully_connected(net, 20)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.fit(training, output, batch_size=4, n_epoch=100, show_metric=True)
    model.save("model")

    return model


def create_m(training, output):
    training = np.array(training)
    output = np.array(output)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    model = Sequential()
    model.add(Dense(128, input_shape=(train_x[0],), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense((train_y[0]), activation="softmax"))

    sgd = SGD(nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=4)
    model.save("weights.h5")
    return model
