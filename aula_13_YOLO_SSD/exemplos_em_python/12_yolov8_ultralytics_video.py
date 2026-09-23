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
cap = cv2.VideoCapture(str(video))
latencias = []
frames = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t0 = time.perf_counter()
    results = model(frame, conf=0.25, iou=0.40, verbose=False)
    latencias.append((time.perf_counter() - t0) * 1000)
    frames += 1

cap.release()

if latencias:
    print(f"Frames processados: {frames}")
    print(f"Latência média YOLOv8n: {sum(latencias)/len(latencias):.2f} ms")
    print(f"FPS aproximado: {1000/(sum(latencias)/len(latencias)):.2f}")

# DESAFIO DO ALUNO:
# Salve um vídeo com as caixas desenhadas usando results[0].plot().
