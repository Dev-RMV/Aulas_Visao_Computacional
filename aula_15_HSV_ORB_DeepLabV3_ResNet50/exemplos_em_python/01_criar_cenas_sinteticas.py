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
from pathlib import Path
from seg_utils import salvar_cenas

ROOT = Path(__file__).resolve().parent
paths = salvar_cenas(ROOT, n=5)

print("Cenas externas sintéticas criadas:")
for p in paths:
    print(" -", p)

# DESAFIO DO ALUNO:
# Abra as imagens e identifique pista, calçada, vegetação, veículos e pedestres.
