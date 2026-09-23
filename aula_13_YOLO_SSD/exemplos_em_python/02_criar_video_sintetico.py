"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

from pathlib import Path
from vision_utils import create_synthetic_video

ROOT = Path(__file__).resolve().parent
video = create_synthetic_video(ROOT / "dados" / "rua_sintetica.avi", n_frames=120)

print("Vídeo sintético criado em:", video)
print("Esse vídeo será usado antes de utilizar webcam ou vídeos reais.")

# DESAFIO DO ALUNO:
# Aumente n_frames para 200 e observe o tamanho do arquivo gerado.
