"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path
from vision_utils import write_csv

ROOT = Path(__file__).resolve().parent
rows = [
    {"modelo": "YOLOv8n", "fps": 0.0, "latencia_ms": 0.0, "tamanho_MB": 0.0, "id_switches_min": 0.0},
    {"modelo": "SSD MobileNetV2", "fps": 0.0, "latencia_ms": 0.0, "tamanho_MB": 0.0, "id_switches_min": 0.0},
]

saida = ROOT / "relatorios" / "metricas_modelos.csv"
write_csv(saida, rows, ["modelo", "fps", "latencia_ms", "tamanho_MB", "id_switches_min"])
print("CSV criado em:", saida)

# DESAFIO DO ALUNO:
# Preencha o CSV com medições reais.
