"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path
import cv2
from vision_utils import create_synthetic_video

ROOT = Path(__file__).resolve().parent
video = create_synthetic_video(ROOT / "dados" / "rua_sintetica.avi", n_frames=30)

cap = cv2.VideoCapture(str(video))
contador = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    contador += 1
    print("Frame", contador, "-> shape:", frame.shape)

cap.release()
print("Total de frames lidos:", contador)

# DESAFIO DO ALUNO:
# Salve o quinto frame como imagem PNG em 'saidas'.
