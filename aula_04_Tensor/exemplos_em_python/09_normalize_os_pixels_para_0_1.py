import tensorflow as tf

bytes_png = tf.io.read_file("assets/carro_real.png")
imagem = tf.io.decode_png(bytes_png, channels=3)
normalizada = tf.cast(imagem, tf.float32) / 255.0

print("antes:", imagem.dtype,
      tf.reduce_min(imagem).numpy(),
      tf.reduce_max(imagem).numpy())
print("depois:", normalizada.dtype,
      tf.reduce_min(normalizada).numpy(),
      tf.reduce_max(normalizada).numpy())
