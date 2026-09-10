from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 27 - Avaliar o modelo Keras no conjunto de teste.
"""
import tensorflow as tf, pandas as pd
IMG_SIZE=(160,160); BATCH=16
def load_sample(path, label):
    img=tf.io.read_file(path); img=tf.image.decode_jpeg(img, channels=3); img=tf.image.resize(img, IMG_SIZE)
    return img, tf.cast(label, tf.float32)
df=pd.read_csv("resultados/21_test.csv")
ds=tf.data.Dataset.from_tensor_slices((df["path"].values, df["gender"].values)).map(load_sample).batch(BATCH)
model=tf.keras.models.load_model("resultados/25_mobilenetv2_genero.keras")
loss, acc = model.evaluate(ds)
print(f"Loss teste: {loss:.4f}")
print(f"Acuracia teste: {acc:.4f}")
