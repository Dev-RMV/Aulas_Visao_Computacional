"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from dnn_utils import list_classification_images, time_ms

model = MobileNetV2(weights='imagenet')
img = cv2.imread(str(list_classification_images()[0]))
rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
x = preprocess_input(rgb.astype('float32'))[None, ...]

media, desvio = time_ms(lambda: model.predict(x, verbose=0), loops=20, warmup=5)
print(f'Latência Keras: {media:.2f} ± {desvio:.2f} ms')

# DESAFIO DO ALUNO:
# Compare CPU e GPU, caso o computador tenha GPU configurada.
