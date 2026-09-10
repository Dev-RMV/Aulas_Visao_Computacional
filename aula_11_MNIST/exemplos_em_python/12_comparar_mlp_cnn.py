"""
Exemplo 12 - Comparação completa do Item A.

Objetivo:
- Treinar MLP e CNN.
- Imprimir model.summary().
- Plotar curvas.
- Comparar parâmetros, tempo por época e acurácia de teste.

Justificativa didática:
A CNN tende a superar o MLP no MNIST porque preserva a estrutura espacial 2D da imagem,
aprende filtros locais úteis para bordas/traços e reutiliza pesos por toda a imagem.
O MLP achata a imagem em um vetor de 784 posições e perde a vizinhança entre pixels.
"""

from mnist_utils import (
    EpochTimer,
    build_cnn,
    build_mlp,
    load_mnist,
    model_summary_row,
    plot_history,
    save_comparison_table,
)

EPOCHS = 5
BATCH = 128

# MLP
(x_train_mlp, y_train), (x_test_mlp, y_test) = load_mnist(flatten=True)
mlp = build_mlp()
print("\n=== RESUMO MLP ===")
mlp.summary()
timer_mlp = EpochTimer()
h_mlp = mlp.fit(x_train_mlp, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH, callbacks=[timer_mlp])
plot_history(h_mlp, "MLP", "resultados/curvas_mlp.png")
_, acc_mlp = mlp.evaluate(x_test_mlp, y_test, verbose=0)

# CNN
(x_train_cnn, y_train), (x_test_cnn, y_test) = load_mnist(flatten=False)
cnn = build_cnn()
print("\n=== RESUMO CNN ===")
cnn.summary()
timer_cnn = EpochTimer()
h_cnn = cnn.fit(x_train_cnn, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH, callbacks=[timer_cnn])
plot_history(h_cnn, "CNN", "resultados/curvas_cnn.png")
_, acc_cnn = cnn.evaluate(x_test_cnn, y_test, verbose=0)

rows = [
    model_summary_row("MLP", mlp, timer_mlp.epoch_times, acc_mlp),
    model_summary_row("CNN", cnn, timer_cnn.epoch_times, acc_cnn),
]
save_comparison_table(rows)

print("\nComentário sobre overfitting:")
print("Procure nas curvas: se accuracy de treino sobe e val_accuracy fica estável/cai, há overfitting.")
print("Também observe se loss de validação começa a subir enquanto loss de treino continua caindo.")
