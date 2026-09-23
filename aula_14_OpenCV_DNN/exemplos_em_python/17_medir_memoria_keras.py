"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import os, psutil, cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from dnn_utils import list_classification_images

process = psutil.Process(os.getpid())
mem_antes = process.memory_info().rss / (1024*1024)
model = MobileNetV2(weights='imagenet')
img = cv2.imread(str(list_classification_images()[0]))
rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
x = preprocess_input(rgb.astype('float32'))[None, ...]
_ = model.predict(x, verbose=0)
mem_depois = process.memory_info().rss / (1024*1024)

print(f'Memória antes:  {mem_antes:.1f} MB')
print(f'Memória depois: {mem_depois:.1f} MB')
print(f'Aumento aproximado: {mem_depois - mem_antes:.1f} MB')

# DESAFIO DO ALUNO:
# Compare com o valor medido no exemplo 16.
