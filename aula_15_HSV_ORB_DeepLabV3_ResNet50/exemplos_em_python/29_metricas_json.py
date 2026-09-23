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

metricas = {
    "calibracao_ms": 2.1,
    "hsv_ms": 1.4,
    "orb_ms": 5.8,
    "detector_classico_ms": 8.2,
    "dnn_ms": 45.0,
    "rastreamento_ms": 3.0,
    "segmentacao_ms": 120.0,
    "observacao": "valores demonstrativos; substituir por medições reais"
}

Path(ROOT / "relatorio" / "metricas_demo.json").write_text(
    json.dumps(metricas, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps(metricas, indent=2, ensure_ascii=False))

# DESAFIO DO ALUNO:
# Gere esse JSON automaticamente a partir dos tempos medidos.
