from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 29 - Salvar modelo e medir tamanho em disco.
"""
from pathlib import Path
from utils.tf_utils import model_size_mb
path=Path("resultados/25_mobilenetv2_genero.keras")
if not path.exists():
    print("Treine primeiro o Exemplo 25.")
else:
    print("Modelo:", path)
    print(f"Tamanho: {model_size_mb(path):.2f} MB")
