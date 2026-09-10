import tensorflow as tf

bytes_png = tf.io.read_file("assets/carro_real.png")
imagem = tf.io.decode_png(bytes_png, channels=3)
lote = tf.expand_dims(imagem, axis=0)

print("imagem:", imagem.shape)
print("lote:", lote.shape)
print("N:", lote.shape[0])
