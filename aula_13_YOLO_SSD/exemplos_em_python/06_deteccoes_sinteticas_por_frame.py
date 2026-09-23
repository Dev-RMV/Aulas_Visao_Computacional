"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from vision_utils import fake_detector

for frame_idx in [0, 10, 30, 50]:
    print("\nFRAME", frame_idx)
    detections = fake_detector(frame_idx)
    for d in detections:
        print(d)

# DESAFIO DO ALUNO:
# Conte quantas detecções aparecem em cada frame antes de aplicar NMS.
