import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 256.0, x_test / 256.0

model = tf.keras.models.Sequential()
model.add( tf.keras.layers.Flatten() )
model.add( tf.keras.layers.Dense(128, activation='relu') )
model.add( tf.keras.layers.Dropout(0.2) )
model.add( tf.keras.layers.Dense(10, activation='softmax') )

class BestModelCB(tf.keras.callbacks.Callback):
  best_val_acc=-1
  def on_epoch_end(self, epoch, logs=None):
    val_acc=logs["val_accuracy"]
    if val_acc>self.best_val_acc:
        self.best_weights = self.model.get_weights()
        self.best_val_acc=val_acc
cb=BestModelCB()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=256,validation_data=(x_test, y_test),callbacks=[cb])

model.evaluate(x_test, y_test)
model.summary()

# Test best Model
model.set_weights(cb.best_weights)
model.evaluate(x_test, y_test)
model.save("best_model")