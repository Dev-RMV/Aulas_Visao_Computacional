"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre segmentação semântica
e integração de técnicas de percepção visual para robótica e veículos autônomos.

A sequência começa com imagens sintéticas e procedimentos clássicos, avança
para modelos profundos pré-treinados e termina com um pipeline integrativo.
Sempre que houver modelo profundo, o código indica a dependência necessária
e a parte que deve ser adaptada para imagens reais.

Ao final há um bloco DESAFIO DO ALUNO para manter uma parte prática da aula.
"""
# Dependências:
# pip install torch torchvision

import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).eval()
classes = weights.meta.get("categories", [])

print("Modelo carregado: FCN-ResNet50")
print("Número de classes:", len(classes))
print("Primeiras classes:", classes[:10])

# DESAFIO DO ALUNO:
# Descubra em qual conjunto de dados essas classes foram treinadas.
