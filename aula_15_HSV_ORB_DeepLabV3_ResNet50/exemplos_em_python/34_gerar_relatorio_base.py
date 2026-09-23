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

ROOT = Path(__file__).resolve().parent
modelo = ROOT / "relatorio" / "RELATORIO_INTEGRATIVO_MODELO.md"

if modelo.exists():
    print(modelo.read_text(encoding="utf-8")[:1000])
else:
    print("Relatório modelo está na pasta relatorio.")

# DESAFIO DO ALUNO:
# Substituir os valores demonstrativos por métricas reais coletadas.
