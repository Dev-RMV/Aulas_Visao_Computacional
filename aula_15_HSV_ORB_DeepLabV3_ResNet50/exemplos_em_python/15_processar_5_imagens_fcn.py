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
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from seg_utils import salvar_cenas

ROOT = Path(__file__).resolve().parent
paths = salvar_cenas(ROOT, 5)

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).eval()
preprocess = weights.transforms()

rng = np.random.default_rng(1)
palette = rng.integers(0, 255, (256, 3), dtype=np.uint8)

for p in paths:
    pil = Image.open(p).convert("RGB")
    x = preprocess(pil).unsqueeze(0)

    with torch.no_grad():
        mask = model(x)["out"][0].argmax(0).byte().cpu().numpy()

    img = cv2.imread(str(p))
    color = cv2.resize(palette[mask], (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 0.65, color, 0.35, 0)

    cv2.imwrite(str(ROOT / "saidas" / f"15_overlay_{p.stem}.png"), overlay)
    print("processado:", p.name)

# DESAFIO DO ALUNO:
# Imprima também as 3 classes com maior área em cada imagem.
