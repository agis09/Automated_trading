import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import model_from_json
import sys

df = pd.read_csv("min_trade.csv")
data = df.loc[:50000-1, "min":"size"].as_matrix().astype('double')

time_len = 300


def make_train_test_data(data, time_len):
    X = np.array([])
    for i in range(len(data) - time_len * 3):
        x = data[i:i + time_len]
        X = np.append(X, x)

    X = np.reshape(X, (len(X) // (time_len * 5), time_len, 5))

    X_test = np.array([])
    for i in range(len(data) - time_len * 3, len(data) - time_len * 2):
        x = data[i:i + time_len]
        X_test = np.append(X_test, x)

    X_test = np.reshape(X_test, (len(X_test) // (time_len * 5), time_len, 5))

    Y = df.loc[time_len * 2:len(data) - time_len - 1, "min":"size"].as_matrix().astype('double')
    Y_test = df.loc[len(data) - time_len: len(data) - 1, "min":"size"].as_matrix().astype('double')

    return X, Y, X_test, Y_test


X,Y,X_test,Y_test = make_train_test_data(data, time_len)

np.save('X_train.npy', X)
np.save('X_test.npy', X_test)
np.save('Y_train.npy', Y)
np.save('Y_test.npy', Y_test)


def make_labels(X, Y, border=4000):
    labels = np.array([])
    cnt_p = 0
    cnt_m = 0
    for i, j in zip(X, Y):
        if i[299][3] + border < j[2] and i[299][3] + border < j[3]:
            for k in range(5):
                labels = np.append(labels, np.array([1, 0, 0]))
            cnt_p += 1
        elif i[299][3] - border > j[2] and i[299][3] - border > j[3]:
            for k in range(5):
                labels = np.append(labels, np.array([0, 0, 1]))
            cnt_m += 1
        else:
            for k in range(5):
                labels = np.append(labels, np.array([0, 1, 0]))
    labels = np.reshape(labels, (len(Y), 5, 3))
    print(cnt_p, cnt_m, len(Y) - (cnt_p + cnt_m))
    return labels


np.save('labels_train', make_labels(X, Y))
np.save('labels_test', make_labels(X_test, Y_test))


X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
Y_train = np.load("labels_train.npy")
Y_test = np.load("labels_test.npy")

n_in = len(X_train[0][0])
n_out = len(Y_train[0])

print(n_in,n_out)


def model_building():
    model = Sequential()
    model.add(GRU(500, input_shape=(time_len, n_in), return_sequences=True))
    model.add(GRU(500, input_shape=(time_len, n_in), return_sequences=False))

    model.add(Dense(n_out))
    model.add(Activation('softmax'))
    optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


model = model_building()

model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          batch_size=10,
          epochs=200)

predict=model.predict(X_test)


json_string = model.to_json()
open('train_GRU2.json', 'w').write(json_string)
model.save_weights("train_GRU2_param.hdf5")


y1,Y1,Y2,y2 = np.split(Y_test,[2,3,4],axis=1 )
p1,P1,P2,p2 = np.split(predict,[2,3,4],axis=1)

plt.plot(Y1, label='actual')
plt.plot(P1, label='predict')
plt.legend(loc='upper left')
plt.savefig("result_first.png")
plt.clf()

plt.plot(Y2, label='actual')
plt.plot(P2, label='predict')
plt.legend(loc='upper left')
plt.savefig("result_last.png")








