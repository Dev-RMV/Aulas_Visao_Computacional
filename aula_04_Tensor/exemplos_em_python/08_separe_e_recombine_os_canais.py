import tensorflow as tf

bytes_png = tf.io.read_file("assets/carro_real.png")
rgb = tf.io.decode_png(bytes_png, channels=3)

r, g, b = tf.split(rgb, 3, axis=-1)
reconstruida = tf.concat([r, g, b], axis=-1)

print(r.shape, g.shape, b.shape)
print("igual:", tf.reduce_all(rgb == reconstruida).numpy())
