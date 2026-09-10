from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 28 - Matriz de confusao para classificacao binaria de genero.
"""
import tensorflow as tf, pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix, classification_report
IMG_SIZE=(160,160)
def load_img(path):
    img=tf.io.read_file(path); img=tf.image.decode_jpeg(img, channels=3); img=tf.image.resize(img, IMG_SIZE)
    return img
df=pd.read_csv("resultados/21_test.csv")
model=tf.keras.models.load_model("resultados/25_mobilenetv2_genero.keras")
imgs=tf.stack([load_img(p) for p in df["path"].values])
prob=model.predict(imgs, verbose=0).ravel()
pred=(prob>=0.5).astype(int)
print(confusion_matrix(df["gender"].values, pred))
print(classification_report(df["gender"].values, pred, target_names=["Masculino", "Feminino"]))
