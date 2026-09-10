import tensorflow as tf

r = tf.fill((4, 6, 1), 255)
g = tf.zeros((4, 6, 1), dtype=tf.int32)
b = tf.zeros((4, 6, 1), dtype=tf.int32)

imagem = tf.concat([r, g, b], axis=-1)
imagem = tf.cast(imagem, tf.uint8)

print("shape:", imagem.shape)
print("pixel [0,0]:", imagem[0, 0].numpy())
