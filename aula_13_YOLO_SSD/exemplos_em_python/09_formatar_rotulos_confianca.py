"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from vision_utils import fake_detector

for d in fake_detector(15, duplicate=False):
    label = f"{d['cls']} | {d['conf']*100:.1f}%"
    print(label)

# DESAFIO DO ALUNO:
# Acrescente as coordenadas da bbox no rótulo impresso.
