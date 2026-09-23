"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Template de benchmark: mede latência média e FPS de uma função de inferência.
Aqui a função é simulada; depois o aluno troca por YOLO ou SSD real.
"""
import time
from vision_utils import draw_scene, fake_detector


def inferencia_simulada(frame, idx):
    # Troque esta função por: model(frame) ou net.forward().
    return fake_detector(idx)

latencias = []
frames = 100

for i in range(frames):
    frame = draw_scene(i)
    t0 = time.perf_counter()
    detections = inferencia_simulada(frame, i)
    latencias.append((time.perf_counter() - t0) * 1000)

media = sum(latencias) / len(latencias)
print(f"Latência média: {media:.3f} ms")
print(f"FPS estimado: {1000/media:.2f}")

# DESAFIO DO ALUNO:
# Substitua inferencia_simulada por uma função de inferência real.
