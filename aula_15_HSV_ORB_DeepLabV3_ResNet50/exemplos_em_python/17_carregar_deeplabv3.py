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
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()

print("Modelo carregado: DeepLabV3-ResNet50")
print("Classes:", len(weights.meta.get("categories", [])))

# DESAFIO DO ALUNO:
# Compare visualmente FCN e DeepLabV3 na mesma imagem.
