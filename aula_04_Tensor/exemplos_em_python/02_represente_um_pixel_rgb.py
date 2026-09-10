import tensorflow as tf

pixel = tf.constant([230, 80, 35], dtype=tf.uint8)

print("pixel:", pixel.numpy())
print("shape:", pixel.shape)
print("rank:", tf.rank(pixel).numpy())
print("R:", pixel[0].numpy())
print("G:", pixel[1].numpy())
print("B:", pixel[2].numpy())
