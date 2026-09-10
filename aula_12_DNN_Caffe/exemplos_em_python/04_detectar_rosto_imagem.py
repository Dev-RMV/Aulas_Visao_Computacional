from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 04 - Detectar rostos em uma imagem usando Haar Cascade.
Uso: python exemplos/04_detectar_rosto_imagem.py --imagem data/faces_teste/pessoa.jpg
"""
import argparse, cv2
from utils.face_utils import load_image, detect_faces_haar, draw_label, ensure_dir
parser = argparse.ArgumentParser()
parser.add_argument("--imagem", default="data/faces_teste/pessoa.jpg")
args = parser.parse_args()
img = load_image(args.imagem)
faces = detect_faces_haar(img)
print(f"Rostos detectados: {len(faces)}")
for i, box in enumerate(faces, start=1):
    draw_label(img, box, f"Rosto {i}")
ensure_dir("resultados")
out = "resultados/04_rostos_detectados.jpg"
cv2.imwrite(out, img)
print("Imagem salva em", out)
