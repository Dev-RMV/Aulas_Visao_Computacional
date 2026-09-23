"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from vision_utils import fake_detector, apply_nms

frame_idx = 20
detections = fake_detector(frame_idx, duplicate=True)
filtered = apply_nms(detections, score_thr=0.25, nms_thr=0.40)

print("Antes do NMS:", len(detections), "caixas")
print("Depois do NMS:", len(filtered), "caixas")

for d in filtered:
    print(d)

# DESAFIO DO ALUNO:
# Mude nms_thr para 0.1 e 0.8. O que acontece com a quantidade de caixas?
