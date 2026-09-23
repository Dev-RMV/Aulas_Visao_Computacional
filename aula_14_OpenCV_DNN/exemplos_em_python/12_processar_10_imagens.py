"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import list_classification_images, get_opencv_net, predict_opencv, topk, draw_top3, OUT

paths = list_classification_images()[:10]
net = get_opencv_net()

for p in paths:
    img = cv2.imread(str(p))
    prob = predict_opencv(img, net)
    vis = draw_top3(img, topk(prob, 3))
    out = OUT / f'12_top3_{p.stem}.jpg'
    cv2.imwrite(str(out), vis)
    print('salvo:', out.name)

# DESAFIO DO ALUNO:
# Garanta que as 10 imagens sejam de categorias realmente diferentes.
