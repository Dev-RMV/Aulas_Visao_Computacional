import argparse

import cv2

from utils import RECURSOS, SAIDAS, segmentar_verde


parser = argparse.ArgumentParser(description="Erode + dilate para limpar máscara.")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()

imagem = cv2.imread(str(RECURSOS / "alvo_hsv.png"))
if imagem is None:
    raise FileNotFoundError("Execute primeiro: python gerar_recursos.py")
bruta, limpa = segmentar_verde(imagem)
comparacao = cv2.hconcat([cv2.cvtColor(bruta, cv2.COLOR_GRAY2BGR), cv2.cvtColor(limpa, cv2.COLOR_GRAY2BGR)])
cv2.imwrite(str(SAIDAS / "07_mascara_bruta_vs_limpa.png"), comparacao)
print("Antes:", cv2.countNonZero(bruta), "pixels; depois:", cv2.countNonZero(limpa), "pixels")
print("Erode remove regiões pequenas; dilate recompõe e conecta o alvo remanescente.")

if not args.sem_janelas:
    cv2.imshow("Bruta | erode+dilate", comparacao)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

