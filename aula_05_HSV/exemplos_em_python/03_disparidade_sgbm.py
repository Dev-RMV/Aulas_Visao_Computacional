import argparse

import cv2

from utils import SAIDAS, calcular_disparidade_sgbm, carregar_par, colorir_jet, normalizar_disparidade


parser = argparse.ArgumentParser(description="Mapa de disparidade com StereoSGBM.")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()

esquerda, direita = carregar_par()
disparidade = calcular_disparidade_sgbm(esquerda, direita)
proximidade, _, validos = normalizar_disparidade(disparidade)
jet = colorir_jet(proximidade)
jet[~validos] = 0
cv2.imwrite(str(SAIDAS / "03_disparidade_sgbm_jet.png"), jet)
print(f"Pixels válidos: {validos.sum()} de {validos.size} ({validos.mean():.1%})")

if not args.sem_janelas:
    cv2.imshow("StereoSGBM + COLORMAP_JET", jet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

