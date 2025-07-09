import tensorflow as tf

#prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 256.0, x_test / 256.0

#construct model
model = tf.keras.models.Sequential()
model.add( tf.keras.layers.Flatten() )
model.add( tf.keras.layers.Dense(128, activation='relu') )
model.add( tf.keras.layers.Dropout(0.2) )
model.add( tf.keras.layers.Dense(10, activation='softmax') )

#train and test
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),batch_size=128)
model.summary()
model.evaluate(x_test, y_test)
