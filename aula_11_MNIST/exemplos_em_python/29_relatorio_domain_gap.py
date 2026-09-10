"""
Exemplo 29 - Diagnóstico do domain gap.

Objetivo:
- Imprimir uma discussão orientada para relatório/apresentação.
"""

texto = """
Domain gap no reconhecimento de dígitos
---------------------------------------
O MNIST é limpo, centralizado, com fundo padronizado e resolução fixa de 28x28.
Já uma foto real pode ter sombras, papel amarelado, fundo irregular, perspectiva,
rotação, escala diferente, traço fino/grosso e ruído de câmera.

Consequência:
Um modelo com alta acurácia no teste MNIST pode errar amostras reais, porque a
distribuição de treino não é igual à distribuição de aplicação.

Mitigação:
Data augmentation cria variações artificiais, como rotação e zoom, para tornar
o modelo menos sensível a pequenas mudanças. Porém, se o problema real tiver
iluminação, perspectiva e segmentação ruins, também será necessário melhorar o
pré-processamento e coletar exemplos reais para validação.
"""

print(texto)
