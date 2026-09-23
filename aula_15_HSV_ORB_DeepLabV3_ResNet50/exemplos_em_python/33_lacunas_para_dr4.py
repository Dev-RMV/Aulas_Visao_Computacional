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
lacunas = [
    "fusão entre câmera, LiDAR, GPS e IMU",
    "planejamento de trajetória usando percepção temporal",
    "validação robusta em cenários reais, clima, noite e oclusões"
]

for i, x in enumerate(lacunas, 1):
    print(f"Lacuna {i}: {x}")

# DESAFIO DO ALUNO:
# Relacione cada lacuna com uma competência futura em robótica móvel.
