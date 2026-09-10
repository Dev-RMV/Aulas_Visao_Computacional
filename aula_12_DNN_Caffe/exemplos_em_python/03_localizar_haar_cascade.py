from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 03 - Localizar o Haar Cascade do OpenCV.
O Haar Cascade sera usado para detectar rostos antes da classificacao.
"""
import cv2
from pathlib import Path
path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
print("Arquivo Haar Cascade:", path)
print("Existe?", path.exists())
