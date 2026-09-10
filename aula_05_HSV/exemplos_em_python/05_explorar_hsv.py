import argparse

import cv2

from utils import RECURSOS, SAIDAS


parser = argparse.ArgumentParser(description="Visualiza BGR, HSV e canais H/S/V.")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()

imagem = cv2.imread(str(RECURSOS / "alvo_hsv.png"))
if imagem is None:
    raise FileNotFoundError("Execute primeiro: python gerar_recursos.py")
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
painel = cv2.hconcat(
    [
        cv2.resize(imagem, (320, 240)),
        cv2.cvtColor(cv2.resize(h, (320, 240)), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.resize(s, (320, 240)), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.resize(v, (320, 240)), cv2.COLOR_GRAY2BGR),
    ]
)
cv2.imwrite(str(SAIDAS / "05_canais_hsv.png"), painel)
print("OpenCV usa H em [0,179] e S,V em [0,255].")
print("Range didático do verde: H=35..85, S=90..255, V=70..255")

if not args.sem_janelas:
    cv2.imshow("BGR | H | S | V", painel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

