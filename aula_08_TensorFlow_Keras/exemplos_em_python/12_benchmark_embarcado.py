"""Exemplo 12 — mede latência de inferência CNN e de extração ORB.

Em robótica embarcada, acurácia não é o único critério: latência, memória,
consumo de energia e estabilidade temporal influenciam a escolha do método.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np

from utils import OUTPUTS, image_paths, load_rgb, load_trained_model


paths = image_paths("test")[:120]
images = np.stack([load_rgb(path) for path in paths])
model = load_trained_model()
orb = cv2.ORB_create(nfeatures=180, edgeThreshold=7, patchSize=17, fastThreshold=6)

# Aquecimento evita contar custos de inicialização na medição.
model.predict(images[:8], verbose=0)
start = time.perf_counter()
model.predict(images, batch_size=1, verbose=0)
cnn_ms = 1000 * (time.perf_counter() - start) / len(images)

start = time.perf_counter()
counts = []
for rgb in images:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keypoints, _ = orb.detectAndCompute(gray, None)
    counts.append(len(keypoints))
orb_ms = 1000 * (time.perf_counter() - start) / len(images)

result = {
    "imagens": len(images),
    "cnn_ms_por_imagem_batch_1": cnn_ms,
    "orb_ms_por_imagem": orb_ms,
    "orb_keypoints_medios": float(np.mean(counts)),
    "observacao": "Resultados dependem do hardware, threads, versão das bibliotecas e aquecimento.",
}
with (OUTPUTS / "12_benchmark.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print("Imagens:", len(images))
print(f"CNN, batch=1: {cnn_ms:.2f} ms/imagem")
print(f"ORB: {orb_ms:.2f} ms/imagem")
print(f"ORB keypoints médios: {np.mean(counts):.1f}")
print("Em um robô, repita o benchmark no hardware-alvo.")

