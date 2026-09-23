"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path
import cv2
from vision_utils import draw_scene, fake_detector, apply_nms, draw_detections, IoUTracker, VIDEO_SIZE, FPS_VIDEO

ROOT = Path(__file__).resolve().parent
saida = ROOT / "saidas" / "24_trilhas_30_frames.avi"
writer = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"MJPG"), FPS_VIDEO, VIDEO_SIZE)
tracker = IoUTracker(iou_thr=0.30, trail_len=30)

for i in range(100):
    frame = draw_scene(i)
    dets = apply_nms(fake_detector(i), 0.25, 0.40)
    tracked = tracker.update(dets)
    out = draw_detections(frame, tracked, show_id=True, tracks=tracker.trails)
    writer.write(out)

writer.release()
print("Vídeo com trilhas salvo em:", saida)

# DESAFIO DO ALUNO:
# Mude trail_len para 10 e depois 60. Compare visualmente.
