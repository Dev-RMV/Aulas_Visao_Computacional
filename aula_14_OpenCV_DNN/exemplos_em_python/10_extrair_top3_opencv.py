"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import list_classification_images, get_opencv_net, predict_opencv, topk, load_labels

img = cv2.imread(str(list_classification_images()[0]))
net = get_opencv_net()
prob = predict_opencv(img, net)
labels = load_labels()

for pos, (idx, conf) in enumerate(topk(prob, 3), start=1):
    print(f'{pos}) índice={idx} label={labels[idx]} confiança={conf*100:.2f}%')

# DESAFIO DO ALUNO:
# Altere k=3 para k=5 e observe a diferença.
