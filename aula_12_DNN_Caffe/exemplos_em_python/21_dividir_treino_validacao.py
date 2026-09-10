from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 21 - Dividir UTKFace em treino, validacao e teste.
"""
import argparse, pandas as pd
from sklearn.model_selection import train_test_split
from utils.tf_utils import collect_utkface_paths, parse_utkface_filename
parser = argparse.ArgumentParser(); parser.add_argument("--pasta", default="data/utkface_sample"); parser.add_argument("--limite", type=int, default=1000)
args = parser.parse_args()
rows=[]
for p in collect_utkface_paths(args.pasta, args.limite):
    age, gender = parse_utkface_filename(p); rows.append({"path":str(p), "gender":gender})
df = pd.DataFrame(rows)
train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["gender"])
val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["gender"])
print(len(train), len(val), len(test))
train.to_csv("resultados/21_train.csv", index=False); val.to_csv("resultados/21_val.csv", index=False); test.to_csv("resultados/21_test.csv", index=False)
