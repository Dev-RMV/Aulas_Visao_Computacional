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
import json

ROOT = Path(__file__).resolve().parent
m = json.loads(Path(ROOT / "relatorio" / "metricas_demo.json").read_text(encoding="utf-8"))

linhas = [
    "| Técnica | Tempo médio (ms) | Complexidade | Observação |",
    "|---|---:|---|---|"
]

for k, v in m.items():
    if k.endswith("_ms"):
        linhas.append(f'| {k.replace("_ms", "")} | {v:.1f} | preencher | medir em hardware alvo |')

texto = "\n".join(linhas)
Path(ROOT / "relatorio" / "tabela_metricas.md").write_text(texto, encoding="utf-8")
print(texto)

# DESAFIO DO ALUNO:
# Acrescente acurácia, memória e consumo estimado.
