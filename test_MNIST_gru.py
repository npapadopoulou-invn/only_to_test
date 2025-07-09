import tensorflow as tf

#prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 256.0, x_test / 256.0

# Each MNIST image batch is a tensor of shape (batch_size, 28, 28). 
# Each input sequence will be of size (28, 28)
# (height is treated like time).
input_dim = 28
hidden_units = 32

#construct model
model = tf.keras.Sequential()
model.add(tf.keras.layers.GRU(hidden_units, input_shape=(None, input_dim)) )
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Softmax())

#train and test
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=30)
model.evaluate(x_test, y_test)
model.summary()
