"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import list_classification_images, get_opencv_net, predict_opencv, time_ms

img = cv2.imread(str(list_classification_images()[0]))
net = get_opencv_net()
media, desvio = time_ms(lambda: predict_opencv(img, net), loops=20, warmup=5)

print(f'Latência OpenCV DNN: {media:.2f} ± {desvio:.2f} ms')

# DESAFIO DO ALUNO:
# Repita com loops=100 e compare a estabilidade da média.
