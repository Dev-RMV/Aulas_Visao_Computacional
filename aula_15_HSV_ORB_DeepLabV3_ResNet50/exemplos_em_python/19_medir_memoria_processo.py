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
# Dependência:
# pip install psutil

import os
import psutil

process = psutil.Process(os.getpid())
mem = process.memory_info().rss / (1024 ** 2)

print(f"Memória atual do processo: {mem:.2f} MB")

# DESAFIO DO ALUNO:
# Meça antes e depois de carregar o modelo FCN.
