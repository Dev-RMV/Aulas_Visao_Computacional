from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 07 - Sobrepor rotulos simulados.
Antes de carregar a rede, entenda como desenhar bbox e texto no frame.
"""
import argparse, cv2
from utils.face_utils import load_image, detect_faces_haar, draw_label, ensure_dir
parser = argparse.ArgumentParser(); parser.add_argument("--imagem", default="data/faces_teste/pessoa.jpg")
args = parser.parse_args()
img = load_image(args.imagem)
faces = detect_faces_haar(img)
for box in faces:
    draw_label(img, box, "Genero: ? | Idade: ?")
ensure_dir("resultados")
cv2.imwrite("resultados/07_rotulos_mock.jpg", img)
print("Arquivo salvo em resultados/07_rotulos_mock.jpg")
