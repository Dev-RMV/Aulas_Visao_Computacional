"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import list_classification_images, predict_opencv, topk, draw_top3, OUT

img_path = list_classification_images()[0]
img = cv2.imread(str(img_path))
prob = predict_opencv(img)
vis = draw_top3(img, topk(prob, 3))

saida = OUT / f'11_top3_{img_path.stem}.jpg'
cv2.imwrite(str(saida), vis)
print('Imagem anotada salva em:', saida)

# DESAFIO DO ALUNO:
# Mude a posição do texto para o canto inferior esquerdo.
