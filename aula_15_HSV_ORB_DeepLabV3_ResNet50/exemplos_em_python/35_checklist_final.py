"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre segmentação semântica
e integração de técnicas de percepção visual para robótica e veículos autônomos.

A sequência começa com imagens sintéticas e procedimentos clássicos, avança
para modelos profundos pré-treinados e termina com um pipeline integrativo.
Sempre que houver modelo profundo, o código indica a dependência necessária
e a parte que deve ser adaptada para imagens reais.

Ao final há um bloco DESAFIO DO ALUNO para manter uma parte prática da aula.
"""
checklist = [
    "5 imagens externas processadas",
    "máscara semântica colorida",
    "overlay semitransparente",
    "percentual de área por classe",
    "comparação HSV vs semântica",
    "tempos por etapa",
    "diagrama do pipeline",
    "tabela de métricas",
    "análise 5 W",
    "arquitetura proposta",
    "3 lacunas para DR4"
]

for item in checklist:
    print("[ ]", item)

# DESAFIO DO ALUNO:
# Marque cada item somente após gerar evidência em imagem, tabela ou texto.
