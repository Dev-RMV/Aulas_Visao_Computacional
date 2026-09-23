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
from pathlib import Path
import time
import torch
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from seg_utils import salvar_cenas

ROOT = Path(__file__).resolve().parent
salvar_cenas(ROOT, 5)

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).eval()
preprocess = weights.transforms()

x = preprocess(Image.open(ROOT / "imagens" / "cena_01.png").convert("RGB")).unsqueeze(0)

for _ in range(2):
    with torch.no_grad():
        model(x)

t0 = time.perf_counter()
n = 5

for _ in range(n):
    with torch.no_grad():
        model(x)

lat = (time.perf_counter() - t0) * 1000 / n
print(f"Latência média FCN: {lat:.2f} ms")

# DESAFIO DO ALUNO:
# Rode em CPU e GPU, se disponível, e compare.
