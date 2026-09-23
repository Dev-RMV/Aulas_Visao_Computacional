"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Template para vídeo real com YOLOv8n e rastreamento por IoU.
O aluno deve fornecer um arquivo de vídeo e ter a biblioteca ultralytics instalada.
"""
from pathlib import Path
import time
import cv2
from vision_utils import IoUTracker, draw_detections

try:
    from ultralytics import YOLO
except ImportError:
    print("Instale com: pip install ultralytics")
    raise SystemExit

ROOT = Path(__file__).resolve().parent
video = ROOT / "dados" / "video_real.mp4"
if not video.exists():
    print("Coloque um vídeo em:", video)
    raise SystemExit

model = YOLO("yolov8n.pt")
tracker = IoUTracker(iou_thr=0.30, trail_len=30)
cap = cv2.VideoCapture(str(video))
latencias = []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t0 = time.perf_counter()
    results = model(frame, conf=0.25, iou=0.40, verbose=False)[0]
    dets = []
    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        cls_id = int(b.cls[0])
        dets.append({"cls": model.names[cls_id], "conf": float(b.conf[0]), "bbox": [x1,y1,x2,y2]})
    tracked = tracker.update(dets)
    out = draw_detections(frame, tracked, show_id=True, tracks=tracker.trails)
    latencias.append((time.perf_counter()-t0)*1000)

cap.release()
print(f"Latência média do pipeline: {sum(latencias)/len(latencias):.2f} ms")

# DESAFIO DO ALUNO:
# Salve o vídeo de saída com as anotações.
