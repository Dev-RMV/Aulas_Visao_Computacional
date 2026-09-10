from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 20 - Criar DataFrame com caminho, idade e genero.
"""
import argparse, pandas as pd
from utils.tf_utils import collect_utkface_paths, parse_utkface_filename
parser = argparse.ArgumentParser(); parser.add_argument("--pasta", default="data/utkface_sample"); parser.add_argument("--limite", type=int, default=1000)
args = parser.parse_args()
rows = []
for p in collect_utkface_paths(args.pasta, args.limite):
    age, gender = parse_utkface_filename(p)
    rows.append({"path": str(p), "age": age, "gender": gender})
df = pd.DataFrame(rows)
print(df.head())
print(df["gender"].value_counts(dropna=False))
df.to_csv("resultados/20_utkface_dataframe.csv", index=False)
