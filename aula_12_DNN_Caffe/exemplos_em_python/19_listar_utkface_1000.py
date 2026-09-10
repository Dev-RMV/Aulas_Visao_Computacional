from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 19 - Listar ate 1000 imagens do UTKFace.
Coloque o dataset em data/utkface_sample ou informe --pasta.
"""
import argparse
from utils.tf_utils import collect_utkface_paths, parse_utkface_filename
parser = argparse.ArgumentParser(); parser.add_argument("--pasta", default="data/utkface_sample"); parser.add_argument("--limite", type=int, default=1000)
args = parser.parse_args()
paths = collect_utkface_paths(args.pasta, args.limite)
print("Imagens validas:", len(paths))
for p in paths[:5]:
    print(p.name, parse_utkface_filename(p))
