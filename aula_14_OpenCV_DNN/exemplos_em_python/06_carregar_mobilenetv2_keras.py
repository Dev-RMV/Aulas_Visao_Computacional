"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet')
print('Modelo Keras carregado.')
print('Entrada esperada:', model.input_shape)
print('Saída esperada:', model.output_shape)

# DESAFIO DO ALUNO:
# Identifique quantas classes a saída possui e relacione com ImageNet.
