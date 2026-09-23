"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from vision_utils import fake_detector, apply_nms

frame_idx = 30
for thr in [0.1, 0.3, 0.4, 0.6, 0.8]:
    dets = fake_detector(frame_idx, duplicate=True)
    filtradas = apply_nms(dets, score_thr=0.25, nms_thr=thr)
    print(f"NMS threshold={thr:.1f}: {len(filtradas)} caixas")

# DESAFIO DO ALUNO:
# Explique por que threshold muito alto pode deixar caixas duplicadas.
