"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

texto = """
Discussão técnica e ética para contagem por drone:

Um sistema de detecção e rastreamento pode estimar fluxo de pedestres ao detectar
pessoas, atribuir IDs persistentes e contar cruzamentos em regiões de interesse.
Em vigilância urbana, esse uso exige finalidade clara, minimização de dados,
sinalização quando aplicável, proteção contra identificação indevida, auditoria
de vieses, limitação de retenção das imagens e avaliação de impacto.

A contagem agregada é menos invasiva do que identificação individual, mas ainda
pode afetar privacidade se o vídeo permitir reconhecer pessoas, rotinas ou locais
sensíveis. Portanto, recomenda-se processar localmente quando possível, descartar
frames brutos após a contagem e reportar somente estatísticas agregadas.
"""

print(texto)

# DESAFIO DO ALUNO:
# Acrescente um parágrafo sobre risco de falso positivo e falso negativo.
