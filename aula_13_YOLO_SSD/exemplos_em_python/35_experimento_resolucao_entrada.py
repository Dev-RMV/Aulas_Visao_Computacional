"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Reduzir a resolução de entrada tende a aumentar FPS, mas pode prejudicar objetos pequenos.
Este exemplo mostra a parte de redimensionamento que entra antes do detector.
"""
import cv2
from vision_utils import draw_scene

frame = draw_scene(20)
for size in [(320, 320), (416, 416), (640, 640)]:
    resized = cv2.resize(frame, size)
    print("Entrada para o modelo:", resized.shape)

# DESAFIO DO ALUNO:
# Meça o tempo de resize para cada resolução usando time.perf_counter().
