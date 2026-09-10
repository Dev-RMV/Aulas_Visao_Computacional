from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 10 - Gerar blob 227x227 para as redes Caffe.
O blob aplica redimensionamento, subtracao de media e organiza canais.
"""
import argparse
from utils.face_utils import load_image, make_age_gender_blob
parser = argparse.ArgumentParser(); parser.add_argument("--imagem", default="data/faces_teste/pessoa.jpg")
args = parser.parse_args()
img = load_image(args.imagem)
blob = make_age_gender_blob(img)
print("Shape do blob:", blob.shape)  # esperado: (1, 3, 227, 227)
print("dtype:", blob.dtype, "min:", blob.min(), "max:", blob.max())
