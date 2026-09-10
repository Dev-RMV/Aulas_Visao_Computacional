from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 09 - Carregar age_net e gender_net com OpenCV DNN.
"""
from utils.face_utils import load_caffe_age_gender
age_net, gender_net = load_caffe_age_gender()
print("Modelos carregados com sucesso.")
print("age_net:", type(age_net))
print("gender_net:", type(gender_net))
