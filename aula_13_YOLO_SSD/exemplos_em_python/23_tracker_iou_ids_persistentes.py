"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from vision_utils import fake_detector, apply_nms, IoUTracker

tracker = IoUTracker(iou_thr=0.30)

for frame_idx in range(0, 15):
    dets = apply_nms(fake_detector(frame_idx), 0.25, 0.40)
    tracked = tracker.update(dets)
    print("\nFrame", frame_idx)
    for t in tracked:
        print("ID", t["id"], t["cls"], t["bbox"])

# DESAFIO DO ALUNO:
# Diminua iou_thr para 0.1 e depois aumente para 0.7. O que muda?
