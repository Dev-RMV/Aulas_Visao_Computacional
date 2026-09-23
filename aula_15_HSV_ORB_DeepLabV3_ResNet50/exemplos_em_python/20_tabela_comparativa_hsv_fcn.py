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
from tabulate import tabulate

linhas = [
    ["HSV por cor", "baixo", "muito alta", "baixo", "falha com iluminação/cor parecida"],
    ["FCN/DeepLab", "alto", "média/baixa em CPU", "alto", "entende contexto e múltiplas classes"],
]

print(tabulate(
    linhas,
    headers=["Técnica", "complexidade", "velocidade", "memória", "observação"],
    tablefmt="github"
))

# DESAFIO DO ALUNO:
# Substitua velocidade e memória por valores medidos na máquina usada.
