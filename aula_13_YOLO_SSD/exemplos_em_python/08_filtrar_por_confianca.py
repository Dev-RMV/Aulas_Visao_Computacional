"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from vision_utils import fake_detector

threshold = 0.80
detections = fake_detector(42, duplicate=True)
filtered = [d for d in detections if d["conf"] >= threshold]

print("Threshold de confiança:", threshold)
print("Detecções originais:", len(detections))
print("Detecções filtradas:", len(filtered))

for d in filtered:
    print(d["cls"], d["conf"])

# DESAFIO DO ALUNO:
# Teste thresholds 0.5, 0.7 e 0.9 e compare os resultados.
