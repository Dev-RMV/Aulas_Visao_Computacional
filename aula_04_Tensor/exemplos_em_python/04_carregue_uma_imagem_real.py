import tensorflow as tf

caminho = "assets/carro_real.png"
bytes_png = tf.io.read_file(caminho)
imagem = tf.io.decode_png(bytes_png, channels=3)

print("shape:", imagem.shape)
print("dtype:", imagem.dtype)
print("primeiro pixel:", imagem[0, 0].numpy())
