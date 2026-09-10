"""Exemplo 5 — mede duas configurações Haar no vídeo com gabarito conhecido."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from utils import RESOURCES, ensure_outputs, load_ground_truth, load_haar


CONFIGS = {
    # Maior sensibilidade: passos menores na escala e menor consenso.
    "sensivel": dict(scaleFactor=1.05, minNeighbors=3, minSize=(35, 35)),
    # Maior especificidade: menos escalas e mais vizinhos exigidos.
    "especifica": dict(scaleFactor=1.20, minNeighbors=6, minSize=(50, 50)),
}


def evaluate(name: str, params: dict) -> dict:
    detector = load_haar()
    cap = cv2.VideoCapture(str(RESOURCES / "video_teste.mp4"))
    truth = load_ground_truth()
    frame_idx = expected_total = detected_total = matched_total = fp_total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = len(detector.detectMultiScale(gray, **params))
        expected = truth[frame_idx]
        detected_total += found
        expected_total += expected
        matched_total += min(found, expected)
        fp_total += max(0, found - expected)
        frame_idx += 1
    cap.release()

    return {
        "configuracao": name,
        "scaleFactor": params["scaleFactor"],
        "minNeighbors": params["minNeighbors"],
        "frames": frame_idx,
        "rostos_esperados": expected_total,
        "deteccoes": detected_total,
        "falsos_positivos_visiveis_estimados": fp_total,
        "taxa_deteccao_estimada_pct": round(100 * matched_total / expected_total, 1),
    }


results = [evaluate(name, params) for name, params in CONFIGS.items()]
output = ensure_outputs() / "05_comparacao_parametros.csv"
with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nCOMPARAÇÃO CONTROLADA")
print("config       scale  vizinhos  detecção estimada  falsos positivos visíveis")
for r in results:
    print(f"{r['configuracao']:<12} {r['scaleFactor']:<6} {r['minNeighbors']:<9} "
          f"{r['taxa_deteccao_estimada_pct']:>7.1f}% {r['falsos_positivos_visiveis_estimados']:>12}")
print(f"\nCSV: {output}")
