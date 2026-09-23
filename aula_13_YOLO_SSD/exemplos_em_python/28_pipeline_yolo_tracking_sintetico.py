"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path
import time
import cv2
from vision_utils import draw_scene, fake_detector, apply_nms, draw_detections, IoUTracker, VIDEO_SIZE, FPS_VIDEO

ROOT = Path(__file__).resolve().parent
saida = ROOT / "saidas" / "28_pipeline_tracking_sintetico.avi"
writer = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"MJPG"), FPS_VIDEO, VIDEO_SIZE)
tracker = IoUTracker(iou_thr=0.30, trail_len=30)
latencias = []

for i in range(120):
    t0 = time.perf_counter()
    frame = draw_scene(i)
    dets = fake_detector(i)                 # substituível por YOLO real
    dets = apply_nms(dets, 0.25, 0.40)       # NMS com threshold 0.4
    tracked = tracker.update(dets)           # IDs persistentes
    out = draw_detections(frame, tracked, show_id=True, tracks=tracker.trails)
    latencias.append((time.perf_counter() - t0) * 1000)
    writer.write(out)

writer.release()
print("Vídeo final salvo em:", saida)
print(f"Latência total média: {sum(latencias)/len(latencias):.2f} ms")
print(f"FPS médio estimado: {1000/(sum(latencias)/len(latencias)):.2f}")

# DESAFIO DO ALUNO:
# Troque fake_detector por uma chamada real do YOLOv8n.
