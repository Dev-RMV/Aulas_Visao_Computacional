"""
Exemplo 16 - Demonstração de overfitting com poucos dados.

Objetivo:
- Forçar uma situação em que o modelo memoriza o treino.
"""

from mnist_utils import build_cnn, load_mnist, plot_history

(x_train, y_train), _ = load_mnist(flatten=False)

# Usar poucas amostras torna a rede mais propensa a memorizar.
x_small = x_train[:1000]
y_small = y_train[:1000]

model = build_cnn()
history = model.fit(x_small, y_small, validation_split=0.3, epochs=12, batch_size=64)
plot_history(history, "CNN com poucos dados", "resultados/overfitting_poucos_dados.png")

print("Indício de overfitting: treino sobe muito, validação não acompanha ou piora.")
