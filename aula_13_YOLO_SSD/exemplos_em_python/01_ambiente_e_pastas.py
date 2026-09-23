"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path
from vision_utils import ensure_dirs

ROOT = Path(__file__).resolve().parent
ensure_dirs(ROOT)

print("Estrutura preparada:")
for pasta in ["dados", "saidas", "modelos", "relatorios"]:
    print(" -", ROOT / pasta)

print("\nUse 'dados' para vídeos/imagens, 'modelos' para pesos e 'saidas' para resultados.")

# DESAFIO DO ALUNO:
# Crie uma subpasta chamada 'testes' dentro de 'saidas'.
