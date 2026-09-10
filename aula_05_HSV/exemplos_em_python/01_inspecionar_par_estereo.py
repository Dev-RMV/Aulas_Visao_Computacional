import argparse

import cv2

from utils import SAIDAS, carregar_par


parser = argparse.ArgumentParser(description="Inspeciona um par estéreo controlado.")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()

esquerda, direita = carregar_par()
print("Esquerda:", esquerda.shape, esquerda.dtype)
print("Direita :", direita.shape, direita.dtype)
par = cv2.hconcat([esquerda, direita])
cv2.imwrite(str(SAIDAS / "01_par_estereo.png"), par)

if not args.sem_janelas:
    cv2.imshow("Par estereo: esquerda | direita", par)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

