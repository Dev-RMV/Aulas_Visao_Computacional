from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 30 - Medir latencia de inferencia do modelo Keras por imagem.
"""
import tensorflow as tf, pandas as pd
from utils.tf_utils import measure_keras_latency_ms
IMG_SIZE=(160,160)
def load_img(path):
    img=tf.io.read_file(path); img=tf.image.decode_jpeg(img, channels=3); img=tf.image.resize(img, IMG_SIZE)
    return img.numpy()
df=pd.read_csv("resultados/21_test.csv")
model=tf.keras.models.load_model("resultados/25_mobilenetv2_genero.keras")
sample=load_img(df.iloc[0]["path"])
lat=measure_keras_latency_ms(model, sample, repeat=30)
print(f"Latencia MobileNetV2 fine-tuned: {lat:.2f} ms/imagem")
