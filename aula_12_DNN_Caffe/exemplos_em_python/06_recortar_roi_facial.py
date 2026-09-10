from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 06 - Recortar ROIs faciais detectadas.
A ROI facial e a entrada para os modelos de genero e idade.
"""
import argparse, cv2
from utils.face_utils import load_image, detect_faces_haar, crop_face, ensure_dir
parser = argparse.ArgumentParser(); parser.add_argument("--imagem", default="data/faces_teste/pessoa.jpg")
args = parser.parse_args()
img = load_image(args.imagem)
faces = detect_faces_haar(img)
ensure_dir("resultados/rois")
for i, box in enumerate(faces):
    roi = crop_face(img, box, padding=20)
    out = f"resultados/rois/face_{i:02d}.jpg"
    cv2.imwrite(out, roi)
    print("ROI salva:", out, roi.shape)
