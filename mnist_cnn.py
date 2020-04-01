import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

dataset = tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test) = dataset.load_data()
plt.imshow(x_test[0])
plt.show()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_acc, val_loss)

# model.save('mnist_prediction_model.model')
# mnist_prediction = tf.keras.models.load_model('mnist_prediction_model.model')
prediction = model.predict(x_test)

max_prob = np.argmax(prediction[1])
print(max_prob)


