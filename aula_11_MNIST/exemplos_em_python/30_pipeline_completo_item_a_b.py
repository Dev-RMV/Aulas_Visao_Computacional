"""
Exemplo 30 - Pipeline completo para o exercício.

Este script reúne Item A e Item B:
- treina MLP e CNN;
- plota curvas;
- imprime tabela comparativa;
- avalia CNN em imagens reais;
- treina CNN com augmentation;
- compara acurácia real antes/depois do augmentation.

Observação didática:
A CNN geralmente supera o MLP porque usa convoluções para explorar relações locais
entre pixels e compartilhar pesos. O MLP recebe a imagem achatada, perdendo a
organização espacial original. Indícios de overfitting aparecem quando a curva de
treino continua melhorando, mas a curva de validação estagna ou piora.
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mnist_utils import (
    EpochTimer,
    build_cnn,
    build_mlp,
    list_real_digit_images,
    load_mnist,
    model_summary_row,
    plot_history,
    predict_real_digit,
    save_comparison_table,
)

EPOCHS = 5
BATCH = 128
REAL_FOLDER = Path("data/real_digits")
Path("resultados").mkdir(exist_ok=True)

# =========================
# Item A - MLP vs CNN
# =========================
(x_train_mlp, y_train), (x_test_mlp, y_test) = load_mnist(flatten=True)
mlp = build_mlp()
print("\n=== MLP ===")
mlp.summary()
t_mlp = EpochTimer()
h_mlp = mlp.fit(x_train_mlp, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH, callbacks=[t_mlp])
plot_history(h_mlp, "MLP", "resultados/pipeline_mlp_curvas.png")
_, acc_mlp = mlp.evaluate(x_test_mlp, y_test, verbose=0)

(x_train_cnn, y_train), (x_test_cnn, y_test) = load_mnist(flatten=False)
cnn = build_cnn()
print("\n=== CNN ===")
cnn.summary()
t_cnn = EpochTimer()
h_cnn = cnn.fit(x_train_cnn, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH, callbacks=[t_cnn])
plot_history(h_cnn, "CNN", "resultados/pipeline_cnn_curvas.png")
_, acc_cnn = cnn.evaluate(x_test_cnn, y_test, verbose=0)
cnn.save("resultados/cnn_mnist.keras")

rows = [
    model_summary_row("MLP", mlp, t_mlp.epoch_times, acc_mlp),
    model_summary_row("CNN", cnn, t_cnn.epoch_times, acc_cnn),
]
save_comparison_table(rows, "resultados/pipeline_comparacao.csv")

# =========================
# Item B - Imagens reais
# =========================
samples = list_real_digit_images(REAL_FOLDER)
if not samples:
    print("\nItem B: nenhuma imagem real encontrada.")
    print("Salve arquivos 0.jpg ... 9.jpg em data/real_digits e execute novamente.")
else:
    correct_clean = 0
    for path, label in samples:
        pred = predict_real_digit(cnn, path)["prediction"]
        correct_clean += int(pred == label)
    acc_real_clean = correct_clean / len(samples)
    print(f"\nAcurácia real antes do augmentation: {acc_real_clean:.2%}")

    datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.10, validation_split=0.1)
    cnn_aug = build_cnn()
    print("\n=== CNN com data augmentation ===")
    cnn_aug.summary()
    h_aug = cnn_aug.fit(
        datagen.flow(x_train_cnn, y_train, batch_size=BATCH, subset="training"),
        validation_data=datagen.flow(x_train_cnn, y_train, batch_size=BATCH, subset="validation"),
        epochs=EPOCHS,
    )
    plot_history(h_aug, "CNN + Augmentation", "resultados/pipeline_cnn_aug_curvas.png")
    cnn_aug.save("resultados/cnn_mnist_aug.keras")

    correct_aug = 0
    for path, label in samples:
        pred = predict_real_digit(cnn_aug, path)["prediction"]
        correct_aug += int(pred == label)
    acc_real_aug = correct_aug / len(samples)
    print(f"Acurácia real após augmentation: {acc_real_aug:.2%}")

    print("\nDiscussão de domain gap:")
    print("MNIST é limpo e centralizado; fotos reais têm sombra, fundo, rotação, escala e ruído.")
    print("Augmentation ajuda porque simula parte dessas variações durante o treino.")
