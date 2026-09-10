import argparse

import cv2
import numpy as np

from utils import SAIDAS, carregar_par, colorir_jet


parser = argparse.ArgumentParser(description="Mapa de disparidade com StereoBM.")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()

esquerda, direita = carregar_par()
g_e = cv2.cvtColor(esquerda, cv2.COLOR_BGR2GRAY)
g_d = cv2.cvtColor(direita, cv2.COLOR_BGR2GRAY)
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
disparidade = stereo.compute(g_e, g_d).astype(np.float32) / 16.0
disp_vis = cv2.normalize(disparidade, None, 0, 1, cv2.NORM_MINMAX)
jet = colorir_jet(disp_vis)
cv2.imwrite(str(SAIDAS / "02_disparidade_bm_jet.png"), jet)
print(f"Disparidade BM: min={disparidade.min():.2f}, max={disparidade.max():.2f}")

if not args.sem_janelas:
    cv2.imshow("StereoBM + COLORMAP_JET", jet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

