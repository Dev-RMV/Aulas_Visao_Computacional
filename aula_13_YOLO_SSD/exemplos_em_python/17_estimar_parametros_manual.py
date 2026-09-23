"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Nem todo arquivo permite contar parâmetros facilmente via OpenCV DNN.
Neste exemplo usamos uma tabela manual que deve ser verificada conforme a variante.
"""
modelos = [
    {"modelo": "YOLOv8n", "parametros_milhoes": 3.2, "fonte": "documentação Ultralytics"},
    {"modelo": "YOLOv4-tiny", "parametros_milhoes": None, "fonte": "preencher conforme arquivo usado"},
    {"modelo": "SSD MobileNetV2", "parametros_milhoes": None, "fonte": "preencher conforme modelo baixado"},
]

for m in modelos:
    print(m)

# DESAFIO DO ALUNO:
# Complete os parâmetros do modelo SSD escolhido e registre a fonte usada.
