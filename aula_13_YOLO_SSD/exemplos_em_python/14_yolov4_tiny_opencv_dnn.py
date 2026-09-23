"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Exemplo clássico com YOLOv4-tiny em Darknet via OpenCV DNN.
Arquivos esperados em modelos/:
- yolov4-tiny.cfg
- yolov4-tiny.weights
- coco.names
"""
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
cfg = ROOT / "modelos" / "yolov4-tiny.cfg"
weights = ROOT / "modelos" / "yolov4-tiny.weights"
names = ROOT / "modelos" / "coco.names"

if not (cfg.exists() and weights.exists() and names.exists()):
    print("Coloque cfg, weights e coco.names na pasta modelos.")
    raise SystemExit

classes = names.read_text(encoding="utf-8").strip().splitlines()
net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))

print("Modelo YOLOv4-tiny carregado.")
print("Número de classes:", len(classes))
print("Camadas de saída:", net.getUnconnectedOutLayersNames())

# DESAFIO DO ALUNO:
# Faça forward em uma imagem usando blobFromImage com tamanho 416x416.
