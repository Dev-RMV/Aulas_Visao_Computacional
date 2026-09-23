"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

import time
from vision_utils import draw_scene, fake_detector

n_frames = 80
t0 = time.perf_counter()
latencias = []

for i in range(n_frames):
    inicio = time.perf_counter()
    frame = draw_scene(i)
    detections = fake_detector(i)
    fim = time.perf_counter()
    latencias.append((fim - inicio) * 1000)

total = time.perf_counter() - t0
fps = n_frames / total

print(f"FPS médio: {fps:.2f}")
print(f"Latência média por frame: {sum(latencias)/len(latencias):.2f} ms")

# DESAFIO DO ALUNO:
# Imprima também a menor e a maior latência observada.
