from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 25 - Treinar apenas a cabeca por ate 10 epocas com EarlyStopping.
"""
import tensorflow as tf, pandas as pd, time
from utils.tf_utils import build_mobilenetv2_gender
IMG_SIZE=(160,160); BATCH=16
def load_sample(path, label):
    img=tf.io.read_file(path); img=tf.image.decode_jpeg(img, channels=3); img=tf.image.resize(img, IMG_SIZE)
    return img, tf.cast(label, tf.float32)
def make_ds(csv, shuffle=False):
    df=pd.read_csv(csv); ds=tf.data.Dataset.from_tensor_slices((df["path"].values, df["gender"].values))
    if shuffle: ds=ds.shuffle(len(df), seed=42)
    return ds.map(load_sample).batch(BATCH).prefetch(tf.data.AUTOTUNE)
train_ds=make_ds("resultados/21_train.csv", True); val_ds=make_ds("resultados/21_val.csv")
model=build_mobilenetv2_gender((160,160,3), 0.3)
early=tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
t0=time.perf_counter()
hist=model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[early])
train_time=time.perf_counter()-t0
model.save("resultados/25_mobilenetv2_genero.keras")
print(f"Tempo total de treino: {train_time:.2f} s")
