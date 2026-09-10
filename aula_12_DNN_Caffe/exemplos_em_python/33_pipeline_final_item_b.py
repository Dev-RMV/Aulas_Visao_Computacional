from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 33 - Pipeline final do Item B.
Executa as etapas principais em sequencia para treino da cabeca e comparacao.
"""
import subprocess, sys
scripts = [
    "19_listar_utkface_1000.py",
    "20_criar_dataframe_utkface.py",
    "21_dividir_treino_validacao.py",
    "24_modelo_mobilenetv2_cabeca.py",
    "25_treinar_cabeca_10_epocas.py",
    "27_avaliar_modelo_keras.py",
    "29_salvar_modelo_e_tamanho.py",
    "30_latencia_modelo_keras.py",
    "31_comparar_caffe_finetuned.py",
]
for s in scripts:
    print("\n=== Executando", s, "===")
    subprocess.run([sys.executable, "exemplos/" + s], check=True)
