import argparse

import cv2
import numpy as np

from utils import SAIDAS, calcular_disparidade_sgbm, carregar_par, colorir_jet, normalizar_disparidade


parser = argparse.ArgumentParser(description="Integra segmentação por cor e profundidade relativa.")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()

esquerda, direita = carregar_par()
disparidade = calcular_disparidade_sgbm(esquerda, direita)
proximidade, _, validos = normalizar_disparidade(disparidade)

hsv = cv2.cvtColor(esquerda, cv2.COLOR_BGR2HSV)
# O objeto PERTO do par sintético é vermelho; a faixa vermelha cruza o zero do Hue.
mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([12, 255, 255]))
mask2 = cv2.inRange(hsv, np.array([168, 100, 80]), np.array([179, 255, 255]))
mask = cv2.bitwise_or(mask1, mask2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
roi_valida = (mask > 0) & validos
if not np.any(roi_valida):
    raise RuntimeError("A ROI não possui disparidades válidas.")

mediana = float(np.median(proximidade[roi_valida]))
p10, p90 = np.percentile(proximidade[roi_valida], [10, 90])
print(f"Proximidade mediana da ROI vermelha: {mediana:.3f}")
print(f"Faixa robusta P10..P90: {p10:.3f} .. {p90:.3f}")

mapa = colorir_jet(proximidade)
mapa[~validos] = 0
overlay = esquerda.copy()
overlay[mask > 0] = (0, 255, 255)
resultado = cv2.addWeighted(overlay, 0.35, esquerda, 0.65, 0)
contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contornos:
    x, y, w, h = cv2.boundingRect(max(contornos, key=cv2.contourArea))
    cv2.rectangle(resultado, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.putText(resultado, f"proximidade mediana={mediana:.2f}", (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

painel = cv2.hconcat([resultado, mapa])
cv2.imwrite(str(SAIDAS / "11_roi_mais_disparidade.png"), painel)

if not args.sem_janelas:
    cv2.imshow("Segmentação + disparidade", painel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

