"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from vision_utils import fake_detector, apply_nms, IoUTracker

tracker = IoUTracker(iou_thr=0.30)
linha_x = 480
lado_anterior = {}
entradas = 0
saidas = 0

for i in range(100):
    dets = apply_nms(fake_detector(i), 0.25, 0.40)
    tracked = tracker.update(dets)

    for t in tracked:
        x1, y1, x2, y2 = t["bbox"]
        cx = (x1 + x2) // 2
        lado = "esq" if cx < linha_x else "dir"
        tid = t["id"]
        if tid in lado_anterior and lado_anterior[tid] != lado:
            if lado == "dir":
                entradas += 1
            else:
                saidas += 1
        lado_anterior[tid] = lado

print("Entradas cumulativas:", entradas)
print("Saídas cumulativas:", saidas)

# DESAFIO DO ALUNO:
# Troque a linha vertical por uma linha horizontal e conte cruzamentos.
