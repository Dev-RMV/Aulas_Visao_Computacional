import tensorflow as tf

x = tf.random.uniform((1, 32, 32, 3))

conv = tf.keras.layers.Conv2D(
    filters=8, kernel_size=3,
    padding="same", activation="relu"
)
y = conv(x)

print("entrada:", x.shape)
print("saída:", y.shape)
print("kernel:", conv.kernel.shape)
