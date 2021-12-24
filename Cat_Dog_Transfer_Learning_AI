import tensorflow as tf
import tensorflow_datasets as tfds

d_train, d_valid, d_test = tfds.load("cats_vs_dogs", 
                                     split=["train[:70%]", "train[70%:85%]", "train[85%:100%]"],
                                     as_supervised=True)

print("n_train = %d" % tf.data.experimental.cardinality(d_train))
print("n_valid = %d" % tf.data.experimental.cardinality(d_valid))
print("n_test = %d" % tf.data.experimental.cardinality(d_test))

size = (64,64)
d_train = d_train.map(lambda x, y: (tf.image.resize(x, size), y))
d_valid = d_valid.map(lambda x, y: (tf.image.resize(x, size), y))
d_test = d_test.map(lambda x, y: (tf.image.resize(x, size), y))

d_train = d_train.map(lambda x, y: (float(x) / 255.0, y))
d_valid = d_valid.map(lambda x, y: (float(x) / 255.0, y))
d_test = d_test.map(lambda x, y: (float(x) / 255.0, y))

batch_size = 32
d_train = d_train.cache().batch(batch_size).prefetch(buffer_size=10)
d_valid = d_valid.cache().batch(batch_size).prefetch(buffer_size=10)
d_test = d_test.cache().batch(batch_size).prefetch(buffer_size=10)

base_model = tf.keras.applications.VGG16(weights="imagenet",
                                         input_shape=(64,64,3),
                                         include_top=False)

base_model.trainable = False

inputs = tf.keras.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalMaxPooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(d_train, epochs=20, batch_size=32,
          shuffle=True, validation_data=d_valid)

model.evaluate(d_test, verbose=0)

base_model.trainable = True


model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(d_train, epochs=10, batch_size=32,
          shuffle=True, validation_data=d_valid)

model.evaluate(d_test, verbose=0)
