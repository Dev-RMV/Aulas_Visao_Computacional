"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

import pandas as pd

linhas = [
    {"modelo": "YOLOv8n", "fps": 0.0, "latencia_ms": 0.0, "parametros_M": 3.2, "tamanho_MB": 0.0},
    {"modelo": "SSD MobileNetV2", "fps": 0.0, "latencia_ms": 0.0, "parametros_M": None, "tamanho_MB": 0.0},
]

df = pd.DataFrame(linhas)
print(df.to_string(index=False))

print("\nSubstitua zeros pelos valores medidos no seu computador ou placa embarcada.")

# DESAFIO DO ALUNO:
# Acrescente uma coluna chamada 'observacao_visual'.
