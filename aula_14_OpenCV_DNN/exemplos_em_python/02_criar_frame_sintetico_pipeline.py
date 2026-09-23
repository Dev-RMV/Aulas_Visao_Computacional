"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

from dnn_utils import create_pipeline_frame

path = create_pipeline_frame()
print('Frame sintético criado em:', path)
print('Ele possui uma região vermelha para segmentação HSV e detalhes para ORB.')

# DESAFIO DO ALUNO:
# Abra a imagem e identifique visualmente qual parte será segmentada por cor.
