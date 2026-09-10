import tensorflow as tf

bytes_png = tf.io.read_file("assets/carro_real.png")
rgb = tf.io.decode_png(bytes_png, channels=3)
cinza = tf.image.rgb_to_grayscale(rgb)

print("RGB:", rgb.shape)
print("cinza:", cinza.shape)
print("dtype:", cinza.dtype)
