"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path
import cv2
from vision_utils import draw_scene, synthetic_objects, draw_detections

ROOT = Path(__file__).resolve().parent
frame_idx = 12
frame = draw_scene(frame_idx)
detections = synthetic_objects(frame_idx)
annotated = draw_detections(frame, detections)

saida = ROOT / "saidas" / "04_bounding_boxes.png"
cv2.imwrite(str(saida), annotated)
print("Imagem anotada salva em:", saida)

# DESAFIO DO ALUNO:
# Altere frame_idx para 40 e compare a posição dos objetos.
