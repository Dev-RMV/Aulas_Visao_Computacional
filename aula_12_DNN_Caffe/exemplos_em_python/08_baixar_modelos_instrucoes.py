from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
#https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
#https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel
#https://github.com/spmallick/learnopencv/blob/master/AgeGender/age_deploy.prototxt
#https://github.com/spmallick/learnopencv/blob/master/AgeGender/gender_deploy.prototxt

"""
Exemplo 08 - Conferir modelos Caffe necessarios.
Por tamanho/licenca, o ZIP nao inclui .caffemodel.
Coloque manualmente os quatro arquivos abaixo em models/opencv_age_gender.
"""
from utils.face_utils import required_caffe_files
print("Arquivos esperados:")
for name, path in required_caffe_files().items():
    print(f"{name:12s}: {path} | existe={path.exists()}")
print("\nNomes esperados:")
print("age_deploy.prototxt, age_net.caffemodel, gender_deploy.prototxt, gender_net.caffemodel")
