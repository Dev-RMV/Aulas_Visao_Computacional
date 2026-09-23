"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import list_classification_images, mobilenet_blob_bgr

img = cv2.imread(str(list_classification_images()[0]))
blob = mobilenet_blob_bgr(img)

print('Formato da imagem original:', img.shape)
print('Formato do blob:', blob.shape)
print('dtype:', blob.dtype)
print('menor valor:', blob.min())
print('maior valor:', blob.max())

# blob = lote x canais x altura x largura.
# Para uma imagem MobileNetV2: (1, 3, 224, 224).

# DESAFIO DO ALUNO:
# Explique por que uma imagem BGR do OpenCV precisa de swapRB=True.
