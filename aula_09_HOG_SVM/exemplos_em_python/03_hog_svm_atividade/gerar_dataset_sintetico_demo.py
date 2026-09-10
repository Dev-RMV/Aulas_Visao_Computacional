"""Somente demonstração do pipeline. Não substitui um dataset real para a entrega."""
from pathlib import Path
import cv2,numpy as np
base=Path(__file__).resolve().parents[1]; pos=base/'dados/positivas'; neg=base/'dados/negativas'; rng=np.random.default_rng(42)
for i in range(120):
    img=rng.integers(0,50,(128,64,3),dtype=np.uint8); cv2.circle(img,(32+int(rng.integers(-3,4)),25),10,(220,220,220),-1); cv2.rectangle(img,(23,38),(41,95),(210,210,210),-1); cv2.imwrite(str(pos/f'pos_{i:03d}.png'),img)
for i in range(120):
    img=rng.integers(0,120,(128,64,3),dtype=np.uint8)
    for _ in range(int(rng.integers(2,7))): cv2.line(img,(int(rng.integers(0,64)),int(rng.integers(0,128))),(int(rng.integers(0,64)),int(rng.integers(0,128))),tuple(map(int,rng.integers(0,180,3))),2)
    cv2.imwrite(str(neg/f'neg_{i:03d}.png'),img)
print('Dataset sintético de demonstração criado.')

# Cria também uma cena de teste sintética com dois objetos.
teste=base/'dados/teste'; teste.mkdir(parents=True,exist_ok=True)
scene=rng.integers(0,55,(384,320,3),dtype=np.uint8)
for cx,cy in [(95,165),(225,190)]:
    cv2.circle(scene,(cx,cy-45),12,(225,225,225),-1)
    cv2.rectangle(scene,(cx-11,cy-30),(cx+11,cy+55),(215,215,215),-1)
cv2.imwrite(str(teste/'cena_sintetica.png'),scene)
print('Cena de teste criada em dados/teste/cena_sintetica.png')
