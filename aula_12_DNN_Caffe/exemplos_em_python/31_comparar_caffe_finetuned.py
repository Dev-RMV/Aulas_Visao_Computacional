from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 31 - Tabela comparativa exigida no Item B.
Preencha a acuracia Caffe com o resultado do relatorio de 5 rostos.
"""
import pandas as pd
from pathlib import Path
from utils.tf_utils import model_size_mb
# Valores de exemplo. Substitua com resultados obtidos nos exemplos anteriores.
acc_caffe = 0.80
acc_ft = 0.87
train_time_s = 320.0
keras_model = Path("resultados/25_mobilenetv2_genero.keras")
size_ft = model_size_mb(keras_model) if keras_model.exists() else None
lat_caffe_ms = 35.0
lat_ft_ms = 18.0
rows = [
    {"modelo":"OpenCV Caffe pre-treinado", "acuracia":acc_caffe, "tempo_treino_s":0, "tamanho_MB":"depende dos .caffemodel", "latencia_ms":lat_caffe_ms},
    {"modelo":"MobileNetV2 fine-tuned", "acuracia":acc_ft, "tempo_treino_s":train_time_s, "tamanho_MB":size_ft, "latencia_ms":lat_ft_ms},
]
df=pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("resultados/31_tabela_comparativa.csv", index=False)
