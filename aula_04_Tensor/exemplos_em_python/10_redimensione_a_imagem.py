import tensorflow as tf

bytes_png = tf.io.read_file("assets/carro_real.png")
imagem = tf.io.decode_png(bytes_png, channels=3)
redimensionada = tf.image.resize(imagem, (224, 224))

print("antes:", imagem.shape, imagem.dtype)
print("depois:", redimensionada.shape,
      redimensionada.dtype)
