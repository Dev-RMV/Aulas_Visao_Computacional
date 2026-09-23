"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

from pathlib import Path
import urllib.request
from dnn_utils import MODELS, ensure_dirs

ensure_dirs()
url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
out = MODELS / 'imagenet_labels.txt'

print('Baixando labels ImageNet...')
urllib.request.urlretrieve(url, out)
print('Arquivo salvo em:', out)
print('Primeiras classes:')
print('\n'.join(out.read_text(encoding='utf-8').splitlines()[:10]))

# DESAFIO DO ALUNO:
# Conte quantas classes existem no arquivo baixado.
