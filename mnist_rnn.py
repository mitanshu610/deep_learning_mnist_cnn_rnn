import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = Sequential()
model.add(LSTM(128, input_shape=x_train.shape[1:],activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)

model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,epochs=3,validation_data=(x_test, y_test))
model.save('mnist_prediction_rnn.model')
prediction = model.predict(x_test)
print(np.argmax(prediction[1]))