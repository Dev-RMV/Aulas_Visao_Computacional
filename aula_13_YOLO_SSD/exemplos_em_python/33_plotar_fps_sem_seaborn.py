"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
modelos = ["YOLOv8n", "SSD MobileNetV2"]
fps = [24.5, 31.2]

plt.figure(figsize=(7,4))
plt.bar(modelos, fps)
plt.ylabel("FPS médio")
plt.title("Comparação de FPS")
plt.tight_layout()

saida = ROOT / "relatorios" / "33_grafico_fps.png"
plt.savefig(saida, dpi=150)
print("Gráfico salvo em:", saida)

# DESAFIO DO ALUNO:
# Crie outro gráfico para latência média.
