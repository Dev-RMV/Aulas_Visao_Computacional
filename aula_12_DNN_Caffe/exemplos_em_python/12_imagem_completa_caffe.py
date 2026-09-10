from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 12 - Pipeline completo em imagem: Haar + ROI + Caffe + rotulo.
"""
import argparse, cv2
from utils.face_utils import load_image, detect_faces_haar, crop_face, load_caffe_age_gender, predict_age_gender, draw_label, ensure_dir
parser = argparse.ArgumentParser(); parser.add_argument("--imagem", default="data/faces_teste/grupo.jpg")
args = parser.parse_args()
img = load_image(args.imagem)
age_net, gender_net = load_caffe_age_gender()
faces = detect_faces_haar(img)
print("Rostos:", len(faces))
for box in faces:
    roi = crop_face(img, box)
    pred = predict_age_gender(roi, age_net, gender_net)
    label = f"{pred.gender} {pred.gender_conf:.2f} | {pred.age} {pred.age_conf:.2f}"
    draw_label(img, box, label)
ensure_dir("resultados")
cv2.imwrite("resultados/12_pipeline_imagem.jpg", img)
print("Resultado salvo em resultados/12_pipeline_imagem.jpg")
