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
analise = """
Em um orçamento de 5 W, técnicas clássicas como HSV, ORB e filtros simples
são candidatas naturais para execução contínua. Redes profundas maiores podem
exigir redução de resolução, quantização, execução intermitente ou acelerador
dedicado. A decisão correta depende de latência, consumo, qualidade da
segmentação e risco da aplicação.
"""

print(analise)

# DESAFIO DO ALUNO:
# Transforme o texto em três recomendações objetivas de projeto.
