"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
A escolha do modelo embarcado deve ser guiada por dados medidos no ambiente alvo.
Este exemplo transforma uma tabela simples em uma conclusão textual.
"""
resultados = [
    {"modelo": "YOLOv8n", "fps": 24.5, "latencia_ms": 40.8, "tamanho_MB": 6.3, "observacao": "melhor precisão visual"},
    {"modelo": "SSD MobileNetV2", "fps": 31.2, "latencia_ms": 32.1, "tamanho_MB": 18.0, "observacao": "mais leve no pipeline testado"},
]

melhor_fps = max(resultados, key=lambda x: x["fps"])
menor_lat = min(resultados, key=lambda x: x["latencia_ms"])

print("Modelo com maior FPS:", melhor_fps["modelo"])
print("Modelo com menor latência:", menor_lat["modelo"])
print("\nConclusão técnica inicial:")
print("Em robótica embarcada, prefira o modelo que cumpre o FPS mínimo, mantém estabilidade e cabe na memória disponível.")
print("A decisão final deve considerar precisão, consumo, tamanho em disco, latência e falhas observadas em campo.")

# DESAFIO DO ALUNO:
# Modifique os valores usando medições reais e gere uma conclusão diferente.
