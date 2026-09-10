"""Classificação: imagem -> CNN -> classe.
Detecção: imagem -> backbone convolucional -> classe + bounding box.
YOLO compartilha cálculo convolucional na imagem inteira, em vez de classificar milhares de janelas independentes.
"""
import cv2,numpy as np
img=np.full((420,720,3),245,np.uint8); cv2.rectangle(img,(90,120),(250,330),(0,160,0),3); cv2.rectangle(img,(390,150),(610,310),(255,0,0),3); cv2.imwrite('20_deteccao_conceito.png',img); print(__doc__)
