"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from vision_utils import iou

box_a = [100, 100, 220, 220]
box_b = [150, 150, 260, 260]
box_c = [400, 100, 500, 200]

print("IoU A-B:", iou(box_a, box_b))
print("IoU A-C:", iou(box_a, box_c))

# DESAFIO DO ALUNO:
# Crie uma box D que tenha IoU maior que 0.5 com a box A.
