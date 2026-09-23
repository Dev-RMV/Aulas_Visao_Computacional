"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Este exemplo usa YOLO exportado para ONNX e carregado pelo OpenCV DNN.
É útil para comparar uma execução mais próxima de sistemas embarcados.

Antes de usar:
1. exporte: yolo export model=yolov8n.pt format=onnx imgsz=640
2. coloque o arquivo ONNX em modelos/yolov8n.onnx
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
onnx = ROOT / "modelos" / "yolov8n.onnx"
img_path = ROOT / "dados" / "imagem_teste.jpg"

if not onnx.exists() or not img_path.exists():
    print("Arquivos esperados:")
    print(" -", onnx)
    print(" -", img_path)
    raise SystemExit

net = cv2.dnn.readNetFromONNX(str(onnx))
img = cv2.imread(str(img_path))
blob = cv2.dnn.blobFromImage(img, 1/255.0, (640,640), swapRB=True, crop=False)
net.setInput(blob)
out = net.forward()

print("Saída bruta do modelo:", out.shape)
print("A decodificação pode variar conforme a exportação do YOLO.")

# DESAFIO DO ALUNO:
# Pesquise o formato da saída do ONNX exportado e implemente a decodificação.
