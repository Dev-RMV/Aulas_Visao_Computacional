"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Em rastreamento real, ID switch ocorre quando a identidade de um objeto muda.
Com vídeo sintético simples, usamos uma aproximação: muitos IDs criados para
poucos objetos persistentes indicam instabilidade do rastreador.
"""
from vision_utils import fake_detector, apply_nms, IoUTracker, FPS_VIDEO

tracker = IoUTracker(iou_thr=0.65, max_missing=1)
frames = 120

for i in range(frames):
    dets = apply_nms(fake_detector(i, jitter=8), 0.25, 0.40)
    tracker.update(dets)

duracao_min = frames / FPS_VIDEO / 60
switches_aprox = max(0, tracker.total_created - 4)
taxa = switches_aprox / duracao_min

print("IDs criados:", tracker.total_created)
print("ID switches aproximados:", switches_aprox)
print(f"Taxa aproximada: {taxa:.2f} ID switches/min")

# DESAFIO DO ALUNO:
# Reduza iou_thr para 0.30 e compare a taxa.
