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
# Na primeira execução, os pesos podem ser baixados automaticamente.

from pathlib import Path
import cv2
import torch
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from seg_utils import salvar_cenas

ROOT = Path(__file__).resolve().parent
salvar_cenas(ROOT, 5)

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).eval()
preprocess = weights.transforms()

img_pil = Image.open(ROOT / "imagens" / "cena_01.png").convert("RGB")
x = preprocess(img_pil).unsqueeze(0)

with torch.no_grad():
    out = model(x)["out"][0]

mask = out.argmax(0).byte().cpu().numpy()

cv2.imwrite(str(ROOT / "saidas" / "11_mask_fcn.png"), mask)
print("Máscara bruta do FCN salva.")

# DESAFIO DO ALUNO:
# Rode com uma foto real de rua e compare a máscara.
