"""Exemplo 04 — cria a CNN e imprime sua arquitetura sem treinar."""

from __future__ import annotations

from utils import OUTPUTS, build_cnn, configure, ensure_dirs


configure(); ensure_dirs()
model = build_cnn()
lines = []
model.summary(print_fn=lines.append)
summary = "\n".join(lines)
print(summary)
print("\nEntrada:", model.input_shape)
print("Saída softmax:", model.output_shape)
print("Vetor de features:", model.get_layer("features_dense").output.shape)
(OUTPUTS / "04_resumo_modelo.txt").write_text(summary, encoding="utf-8")

