"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

from dnn_utils import create_synthetic_classification_images

paths = create_synthetic_classification_images(10)
print('Imagens criadas:')
for p in paths:
    print('-', p.name)
print('\nEssas imagens servem para testar o fluxo; para medir acurácia real, use fotos reais.')

# DESAFIO DO ALUNO:
# Substitua as imagens sintéticas por 10 imagens reais de categorias distintas.
