"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path
import cv2
from vision_utils import draw_scene, fake_detector, apply_nms, draw_detections, VIDEO_SIZE, FPS_VIDEO

ROOT = Path(__file__).resolve().parent
saida = ROOT / "saidas" / "10_video_anotado.avi"

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(str(saida), fourcc, FPS_VIDEO, VIDEO_SIZE)

for i in range(100):
    frame = draw_scene(i)
    dets = fake_detector(i)
    dets = apply_nms(dets, score_thr=0.25, nms_thr=0.40)
    out = draw_detections(frame, dets)
    writer.write(out)

writer.release()
print("Vídeo anotado salvo em:", saida)

# DESAFIO DO ALUNO:
# Escreva FPS e número de objetos detectados sobre cada frame.
