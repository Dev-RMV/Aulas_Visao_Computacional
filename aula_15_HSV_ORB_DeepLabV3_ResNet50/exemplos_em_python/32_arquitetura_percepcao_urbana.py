"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre segmentação semântica
e integração de técnicas de percepção visual para robótica e veículos autônomos.

A sequência começa com imagens sintéticas e procedimentos clássicos, avança
para modelos profundos pré-treinados e termina com um pipeline integrativo.
Sempre que houver modelo profundo, o código indica a dependência necessária
e a parte que deve ser adaptada para imagens reais.

Ao final há um bloco DESAFIO DO ALUNO para manter uma parte prática da aula.
"""
arquitetura = [
    "Calibração e undistort",
    "Segmentação HSV para pistas/faixas simples",
    "ORB para pontos de referência",
    "Detector profundo para objetos dinâmicos",
    "Rastreamento por ID para consistência temporal",
    "Segmentação semântica para área navegável"
]

for i, etapa in enumerate(arquitetura, 1):
    print(i, etapa)

# DESAFIO DO ALUNO:
# Indique entrada, saída e métrica de cada etapa.
