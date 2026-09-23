"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import list_classification_images, get_opencv_net, mobilenet_blob_bgr

img = cv2.imread(str(list_classification_images()[0]))
net = get_opencv_net()
blob = mobilenet_blob_bgr(img)

net.setInput(blob)
out = net.forward()

print('Formato da saída:', out.shape)
print('Exemplo dos 10 primeiros valores:')
print(out.reshape(-1)[:10])

# DESAFIO DO ALUNO:
# Explique por que a maior saída indica a classe mais provável.
