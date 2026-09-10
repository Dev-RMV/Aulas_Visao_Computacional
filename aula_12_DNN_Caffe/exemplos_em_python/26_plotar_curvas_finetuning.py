from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 26 - Plotar curvas de acuracia e loss do fine-tuning.
Este script deve ser adaptado caso voce treine no Exemplo 25 e salve o historico.
"""
import json, matplotlib.pyplot as plt
# Exemplo didatico: curvas simuladas quando o arquivo de historico ainda nao existe.
history = {
    "accuracy": [0.60,0.70,0.76,0.80,0.83,0.85],
    "val_accuracy": [0.58,0.67,0.72,0.75,0.74,0.73],
    "loss": [0.68,0.55,0.46,0.40,0.36,0.33],
    "val_loss": [0.69,0.60,0.52,0.48,0.50,0.55]
}
plt.figure(figsize=(7,4)); plt.plot(history["accuracy"], label="treino"); plt.plot(history["val_accuracy"], label="validacao")
plt.title("Acuracia - fine-tuning da cabeca"); plt.xlabel("Epoca"); plt.ylabel("Acuracia"); plt.legend(); plt.tight_layout(); plt.savefig("resultados/26_curva_acc.png", dpi=150)
plt.figure(figsize=(7,4)); plt.plot(history["loss"], label="treino"); plt.plot(history["val_loss"], label="validacao")
plt.title("Loss - fine-tuning da cabeca"); plt.xlabel("Epoca"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.savefig("resultados/26_curva_loss.png", dpi=150)
print("Curvas salvas em resultados/26_curva_acc.png e resultados/26_curva_loss.png")
print("Indicio de overfitting: loss de validacao sobe enquanto loss de treino continua caindo.")
