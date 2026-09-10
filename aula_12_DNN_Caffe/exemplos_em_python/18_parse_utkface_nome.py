from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 18 - Entender o nome dos arquivos UTKFace.
Formato: idade_genero_etnia_data.jpg. Genero: 0 masculino, 1 feminino.
"""
from utils.tf_utils import parse_utkface_filename
amostras = ["25_0_0_20170116174525125.jpg", "31_1_2_201701040202.jpg"]
for nome in amostras:
    idade, genero = parse_utkface_filename(nome)
    print(nome, "-> idade:", idade, "genero:", genero, "label:", "Feminino" if genero == 1 else "Masculino")
