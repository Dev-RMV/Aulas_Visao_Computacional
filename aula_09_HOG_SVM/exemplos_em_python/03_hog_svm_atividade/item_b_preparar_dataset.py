from pathlib import Path
import cv2
base=Path(__file__).resolve().parents[1]; out=base/'dados/processadas'
for nome in ['positivas','negativas']:
    src=base/'dados'/nome; dst=out/nome; dst.mkdir(parents=True,exist_ok=True)
    arqs=[p for p in src.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
    print(nome,'encontradas:',len(arqs))
    for i,p in enumerate(arqs):
        img=cv2.imread(str(p))
        if img is not None: cv2.imwrite(str(dst/f'{i:03d}.png'),cv2.resize(img,(64,128)))
print('Use no mínimo 100 positivas e 100 negativas para atender ao enunciado.')
