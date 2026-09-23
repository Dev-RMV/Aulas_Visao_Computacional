"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import pandas as pd

# Preencha os valores com as medições feitas nos exemplos anteriores.
dados = [
    {'backend':'OpenCV DNN', 'latencia_ms':0.0, 'memoria_MB':0.0, 'top1_acc':0.0},
    {'backend':'Keras',      'latencia_ms':0.0, 'memoria_MB':0.0, 'top1_acc':0.0},
]

df = pd.DataFrame(dados)
print(df.to_string(index=False))
print('\nComentário técnico:')
print('OpenCV DNN tende a ser preferível quando o objetivo é inferência leve,')
print('integração direta com pipeline OpenCV e menor dependência de frameworks completos.')
print('Keras é preferível para treinamento, ajuste fino, experimentação e validação de modelos.')

# DESAFIO DO ALUNO:
# Substitua os zeros pelas medições reais do seu computador.
