"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
modelos = [
    ROOT / "modelos" / "yolov8n.pt",
    ROOT / "modelos" / "yolov8n.onnx",
    ROOT / "modelos" / "yolov4-tiny.weights",
    ROOT / "modelos" / "frozen_inference_graph.pb",
]

for m in modelos:
    if m.exists():
        mb = m.stat().st_size / (1024*1024)
        print(f"{m.name}: {mb:.2f} MB")
    else:
        print(f"{m.name}: arquivo não encontrado")

# DESAFIO DO ALUNO:
# Acrescente uma coluna 'modelo' e salve esses dados em CSV.
