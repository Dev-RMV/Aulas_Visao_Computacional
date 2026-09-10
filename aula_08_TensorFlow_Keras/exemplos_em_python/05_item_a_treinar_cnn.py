"""Item A completo — treina a CNN por pelo menos 10 épocas.

Diagnóstico de overfitting usado após o treino:
- diferença final treino − validação acima de 8 pontos percentuais; ou
- loss de validação final mais de 10% acima do seu mínimo, depois de já ter
  passado pelo menos duas épocas desde o melhor valor.

O diagnóstico é um indício, não uma prova. As curvas devem sempre ser lidas em
conjunto com o tamanho e a dificuldade do dataset.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from utils import (
    MODELS, OUTPUTS, build_cnn, configure, ensure_dirs, load_datasets,
    overfitting_diagnosis,
)


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()
if args.epochs < 10:
    raise ValueError("O requisito pede no mínimo 10 épocas.")

configure(); ensure_dirs()
train_ds, val_ds, test_ds = load_datasets(args.batch_size)
model = build_cnn()

start = time.perf_counter()
history_obj = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    verbose=2,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        )
    ],
)
training_seconds = time.perf_counter() - start
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
model.save(MODELS / "cnn_geometrica.keras")

history = {key: [float(v) for v in values] for key, values in history_obj.history.items()}
diagnosis = overfitting_diagnosis(history)

epochs = np.arange(1, args.epochs + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].plot(epochs, history["accuracy"], "o-", label="treino", color="#3D8DFF")
axes[0].plot(epochs, history["val_accuracy"], "o-", label="validação", color="#EF5350")
axes[0].set(title="Acurácia", xlabel="Época", ylabel="Acurácia", ylim=(0, 1.03))
axes[0].grid(alpha=0.25); axes[0].legend()
axes[1].plot(epochs, history["loss"], "o-", label="treino", color="#3D8DFF")
axes[1].plot(epochs, history["val_loss"], "o-", label="validação", color="#EF5350")
axes[1].set(title="Loss", xlabel="Época", ylabel="Entropia cruzada")
axes[1].grid(alpha=0.25); axes[1].legend()
fig.suptitle("CNN do zero — treino e validação")
fig.tight_layout()
fig.savefig(OUTPUTS / "05_curvas_treino.png", dpi=180, bbox_inches="tight")
plt.close(fig)

summary = {
    "epocas": args.epochs,
    "batch_size": args.batch_size,
    "tempo_treino_s": training_seconds,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "diagnostico_overfitting": diagnosis,
}
with (OUTPUTS / "05_historico.json").open("w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)
with (OUTPUTS / "05_resultado_treino.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n=== ITEM A — RESULTADO FINAL ===")
print(f"Épocas executadas: {args.epochs}")
print(f"Tempo de treinamento: {training_seconds:.2f} s")
print(f"Acurácia de teste: {100 * test_accuracy:.2f}%")
print(f"Loss de teste: {test_loss:.4f}")
print(f"Gap treino-validação: {100 * diagnosis['gap_acuracia_final']:.2f} p.p.")
print("Overfitting:", diagnosis["comentario"])
print("Curvas:", OUTPUTS / "05_curvas_treino.png")
print("Modelo:", MODELS / "cnn_geometrica.keras")

