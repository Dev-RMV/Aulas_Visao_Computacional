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

diagrama = """flowchart LR
A[Calibração] --> B[Pré-processamento]
B --> C[Segmentação HSV]
B --> D[ORB]
D --> E[Detecção clássica]
E --> F[Detecção profunda]
F --> G[Rastreamento]
G --> H[Segmentação semântica]
H --> I[Decisão de percepção]
"""

Path(ROOT / "relatorio" / "pipeline.mmd").write_text(diagrama, encoding="utf-8")
print(diagrama)

# DESAFIO DO ALUNO:
# Acrescente uma etapa de fusão temporal antes da decisão.
