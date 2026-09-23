"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import get_opencv_net

net = get_opencv_net()
print('Modelo carregado no OpenCV DNN.')
print('Camadas da rede:', len(net.getLayerNames()))

# Comentário técnico:
# OpenCV DNN é voltado ao forward pass, isto é, inferência.
# Para treinamento ou fine-tuning, usa-se normalmente TensorFlow/Keras/PyTorch.

# DESAFIO DO ALUNO:
# Imprima os nomes das 10 primeiras camadas.
