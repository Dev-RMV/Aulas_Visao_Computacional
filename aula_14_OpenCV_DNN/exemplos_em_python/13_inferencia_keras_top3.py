"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from dnn_utils import list_classification_images

model = MobileNetV2(weights='imagenet')
img = cv2.imread(str(list_classification_images()[0]))
rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
x = preprocess_input(rgb.astype('float32'))[None, ...]
pred = model.predict(x, verbose=0)

for rank, (_, label, conf) in enumerate(decode_predictions(pred, top=3)[0], start=1):
    print(f'{rank}) {label}: {conf*100:.2f}%')

# DESAFIO DO ALUNO:
# Compare visualmente o top-3 do Keras com o top-3 do OpenCV DNN.
