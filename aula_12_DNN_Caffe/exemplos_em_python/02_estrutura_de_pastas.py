from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 02 - Criar estrutura de pastas.
Execute uma vez para criar data/, models/ e resultados/.
"""
from pathlib import Path
for p in ["data/faces_teste", "data/utkface_sample", "models/opencv_age_gender", "resultados"]:
    Path(p).mkdir(parents=True, exist_ok=True)
    print("OK:", p)
print("Coloque imagens de teste em data/faces_teste e os modelos Caffe em models/opencv_age_gender.")
