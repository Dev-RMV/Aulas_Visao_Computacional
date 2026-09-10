import argparse

import cv2

from utils import RECURSOS, SAIDAS, segmentar_verde


parser = argparse.ArgumentParser(description="Segmentação HSV em imagem estática.")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()

imagem = cv2.imread(str(RECURSOS / "alvo_hsv.png"))
if imagem is None:
    raise FileNotFoundError("Execute primeiro: python gerar_recursos.py")
mascara_bruta, _ = segmentar_verde(imagem)
segmentado = cv2.bitwise_and(imagem, imagem, mask=mascara_bruta)
cv2.imwrite(str(SAIDAS / "06_mascara_hsv_bruta.png"), mascara_bruta)
cv2.imwrite(str(SAIDAS / "06_objeto_segmentado.png"), segmentado)
print(f"Pixels classificados como alvo: {cv2.countNonZero(mascara_bruta)}")

if not args.sem_janelas:
    cv2.imshow("Máscara HSV bruta", mascara_bruta)
    cv2.imshow("Objeto segmentado", segmentado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

