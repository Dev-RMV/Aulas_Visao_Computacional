"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from vision_utils import iou

tracks_anteriores = {
    1: [100, 100, 200, 220],
    2: [400, 160, 520, 230],
}

deteccoes_atuais = [
    [108, 104, 208, 224],
    [392, 162, 512, 232],
]

for tid, box_track in tracks_anteriores.items():
    print("\nTrack ID", tid)
    for i, box_det in enumerate(deteccoes_atuais):
        print(f"  IoU com detecção {i}:", iou(box_track, box_det))

# DESAFIO DO ALUNO:
# Associe manualmente cada detecção ao ID mais provável.
