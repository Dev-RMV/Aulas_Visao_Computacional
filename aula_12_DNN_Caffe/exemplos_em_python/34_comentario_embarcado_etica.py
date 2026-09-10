from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 34 - Comentario tecnico e etico para inserir no notebook.

Quando usar modelo fixo pre-treinado?
- Quando ha pouca ou nenhuma base rotulada local.
- Quando a meta e prototipagem rapida em tempo real.
- Quando o hardware e limitado e nao permite retreino frequente.
- Quando o custo de manutencao precisa ser baixo.

Quando usar fine-tuning?
- Quando o ambiente do robo difere muito do dataset original: camera, iluminacao, angulo, populacao, resolucao.
- Quando ha amostras locais rotuladas com consentimento.
- Quando o erro do modelo pre-treinado e inaceitavel para a aplicacao.
- Quando e possivel validar vieses, desempenho por subgrupos e latencia no hardware real.

Cuidado:
- Gênero inferido por aparencia nao equivale a identidade de genero.
- A idade e estimada por faixa, com alta incerteza.
- Em sistemas roboticos, use esses atributos apenas quando houver justificativa, consentimento e plano de mitigacao de vies.
"""
print(__doc__)
