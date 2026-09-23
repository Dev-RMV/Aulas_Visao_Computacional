"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import os, psutil, cv2
from dnn_utils import list_classification_images, get_opencv_net, predict_opencv

process = psutil.Process(os.getpid())
mem_antes = process.memory_info().rss / (1024*1024)
net = get_opencv_net()
img = cv2.imread(str(list_classification_images()[0]))
_ = predict_opencv(img, net)
mem_depois = process.memory_info().rss / (1024*1024)

print(f'Memória antes:  {mem_antes:.1f} MB')
print(f'Memória depois: {mem_depois:.1f} MB')
print(f'Aumento aproximado: {mem_depois - mem_antes:.1f} MB')

# DESAFIO DO ALUNO:
# Meça novamente depois de várias inferências e observe se a memória estabiliza.
