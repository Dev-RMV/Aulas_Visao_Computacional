"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Este exemplo mostra o caminho prático com YOLOv8n via Ultralytics.
Na primeira execução, a biblioteca pode baixar o arquivo yolov8n.pt.
"""
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Instale com: pip install ultralytics")
    raise SystemExit

ROOT = Path(__file__).resolve().parent
imagem = ROOT / "dados" / "imagem_teste.jpg"

if not imagem.exists():
    print("Coloque uma imagem em:", imagem)
    raise SystemExit

model = YOLO("yolov8n.pt")
results = model(str(imagem), conf=0.25, iou=0.40)
annotated = results[0].plot()

saida = ROOT / "saidas" / "11_yolov8_imagem.jpg"
import cv2
cv2.imwrite(str(saida), annotated)
print("Resultado salvo em:", saida)

# DESAFIO DO ALUNO:
# Teste com 3 imagens diferentes e registre quais classes aparecem.
