"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

from dnn_utils import ensure_dirs, DATA_CLASS, DATA_PIPE, MODELS, OUT

ensure_dirs()

print('Pastas principais:')
print('Imagens de classificação:', DATA_CLASS)
print('Imagem/frame do pipeline:', DATA_PIPE)
print('Modelos:', MODELS)
print('Saídas:', OUT)

# DESAFIO DO ALUNO:
# Coloque pelo menos 10 imagens reais em data/classificacao/.
# As imagens devem ser de categorias diferentes.
