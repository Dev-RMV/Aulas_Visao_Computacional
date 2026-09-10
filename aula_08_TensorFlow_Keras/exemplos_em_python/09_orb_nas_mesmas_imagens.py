"""Exemplo 09 — extrai keypoints e descritores ORB nas mesmas 20 imagens."""

from __future__ import annotations

import csv

import cv2
import numpy as np

from utils import OUTPUTS, PROJECT, ensure_dirs, read_selection


ensure_dirs()
selection = read_selection()
orb = cv2.ORB_create(nfeatures=180, edgeThreshold=7, patchSize=17, fastThreshold=6)
rows, tiles = [], []

for item in selection:
    image = cv2.imread(str(PROJECT / item["path"]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    count = len(keypoints)
    rows.append({
        "arquivo": item["path"],
        "classe": item["classe"],
        "keypoints": count,
        "descritores": 0 if descriptors is None else descriptors.shape[0],
        "dimensao_por_descritor": 0 if descriptors is None else descriptors.shape[1],
    })
    # Pontos de raio 1 deixam a forma visível. DRAW_RICH_KEYPOINTS desenharia
    # círculos grandes de escala e poluiria esta montagem de 20 imagens.
    visual = image.copy()
    for keypoint in keypoints:
        point = tuple(np.rint(keypoint.pt).astype(int))
        cv2.circle(visual, point, 1, (0, 230, 255), -1, cv2.LINE_AA)
    visual = cv2.resize(visual, (160, 160), interpolation=cv2.INTER_NEAREST)
    cv2.putText(visual, f"{item['classe']} | {count} kp", (4, 152),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(visual, f"{item['classe']} | {count} kp", (4, 152),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(visual)

while len(tiles) % 5:
    tiles.append(np.full_like(tiles[0], 245))
grid = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
cv2.imwrite(str(OUTPUTS / "09_orb_keypoints_20_imagens.jpg"), grid)

with (OUTPUTS / "09_orb_estatisticas.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)

print("Imagens processadas:", len(rows))
print(f"Keypoints ORB por imagem: média={np.mean([r['keypoints'] for r in rows]):.1f}, "
      f"mínimo={min(r['keypoints'] for r in rows)}, máximo={max(r['keypoints'] for r in rows)}")
print("Cada descritor ORB possui 32 bytes / 256 bits.")
print("Montagem:", OUTPUTS / "09_orb_keypoints_20_imagens.jpg")
