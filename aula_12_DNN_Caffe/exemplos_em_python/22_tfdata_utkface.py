from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 22 - Criar pipeline tf.data para imagens faciais.
"""
import tensorflow as tf, pandas as pd
IMG_SIZE = (160, 160)
def load_sample(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img, tf.cast(label, tf.float32)

df = pd.read_csv("resultados/21_train.csv")
ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["gender"].values))
ds = ds.map(load_sample).batch(16).prefetch(tf.data.AUTOTUNE)
for x, y in ds.take(1):
    print("Batch imagens:", x.shape)
    print("Batch labels:", y.shape, y[:10].numpy())
