import tensorflow as tf

bytes_png = tf.io.read_file("assets/carro_real.png")
imagem = tf.io.decode_png(bytes_png, channels=3)

print("shape:", imagem.shape)
print("rank:", tf.rank(imagem).numpy())
print("dtype:", imagem.dtype)
print("mínimo:", tf.reduce_min(imagem).numpy())
print("máximo:", tf.reduce_max(imagem).numpy())
