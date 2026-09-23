"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Este template organiza a comparação final entre YOLO e SSD.
Os valores devem vir de medições feitas no mesmo vídeo, mesma resolução e mesma máquina.
"""
import pandas as pd

resultados = [
    {"modelo": "YOLO", "fps": None, "latencia_ms": None, "tamanho_MB": None, "parametros_M": None, "idswitch_min": None},
    {"modelo": "SSD", "fps": None, "latencia_ms": None, "tamanho_MB": None, "parametros_M": None, "idswitch_min": None},
]

df = pd.DataFrame(resultados)
print(df.to_string(index=False))

print("\nRegra de comparação justa:")
print("Use o mesmo vídeo, mesmo tamanho de entrada, mesma máquina e o mesmo critério de confiança/NMS.")

# DESAFIO DO ALUNO:
# Preencha a tabela com os valores obtidos experimentalmente.
