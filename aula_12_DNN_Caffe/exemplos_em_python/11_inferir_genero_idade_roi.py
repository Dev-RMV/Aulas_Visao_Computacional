from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 11 - Predizer genero aparente e faixa etaria de uma ROI facial.
Use uma imagem ja recortada de rosto.
"""
import argparse
from utils.face_utils import load_image, load_caffe_age_gender, predict_age_gender
parser = argparse.ArgumentParser(); parser.add_argument("--imagem", default="resultados/rois/face_00.jpg")
args = parser.parse_args()
face = load_image(args.imagem)
age_net, gender_net = load_caffe_age_gender()
pred = predict_age_gender(face, age_net, gender_net)
print(pred)
