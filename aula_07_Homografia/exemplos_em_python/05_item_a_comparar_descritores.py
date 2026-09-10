"""Item A completo — compara SIFT, ORB e AKAZE.

Tabela comparativa comentada:

Método | Tipo e dimensão padrão       | Distância | Leitura de engenharia
SIFT   | float32, 128 valores         | L2        | robusto a escala/rotação; vetor maior e mais lento
ORB    | uint8, 32 bytes (256 bits)   | Hamming   | muito rápido e compacto; adequado a CPU embarcada
AKAZE  | uint8, 61 bytes (MLDB padrão)| Hamming   | escala não linear; equilíbrio entre robustez e custo

A contagem e o tempo dependem da imagem, hardware, versão do OpenCV e parâmetros.
Por isso a tabela qualitativa acima deve ser acompanhada pela medição impressa.
"""

from __future__ import annotations

import csv

import cv2
import numpy as np

from utils import descriptor_info, ensure_outputs, extract, load_pair


image_a, image_b = load_pair()
rows = []
colors = {"SIFT": (61, 141, 255), "ORB": (49, 196, 108), "AKAZE": (69, 81, 237)}

for method in ("SIFT", "ORB", "AKAZE"):
    kp_a, desc_a, time_a = extract(method, image_a)
    kp_b, desc_b, time_b = extract(method, image_b)
    _, dtype, description = descriptor_info(method)
    rows.append({
        "metodo": method,
        "keypoints_A": len(kp_a),
        "keypoints_B": len(kp_b),
        "tempo_A_ms": round(time_a, 2),
        "tempo_B_ms": round(time_b, 2),
        "dimensao_por_ponto": desc_a.shape[1],
        "dtype": dtype,
        "descricao": description,
    })
    vis_a = cv2.drawKeypoints(image_a, kp_a, None, colors[method], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_b = cv2.drawKeypoints(image_b, kp_b, None, colors[method], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str(ensure_outputs() / f"05_{method.lower()}_keypoints.jpg"), np.hstack([vis_a, vis_b]))

csv_path = ensure_outputs() / "05_tabela_descritores.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nCOMPARAÇÃO DE EXTRAÇÃO — mediana de 7 repetições")
print("método   kp_A   kp_B   tempo_A(ms)  tempo_B(ms)  dimensão  tipo")
for row in rows:
    print(f"{row['metodo']:<8} {row['keypoints_A']:>5} {row['keypoints_B']:>6} "
          f"{row['tempo_A_ms']:>12.2f} {row['tempo_B_ms']:>12.2f} "
          f"{row['dimensao_por_ponto']:>9}  {row['dtype']}")
print(f"\nCSV: {csv_path}")
