import cv2
import numpy as np

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Backend de vídeo disponível. Teste rápido de câmera 0...")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ok, frame = cap.read()
    if ok:
        print("Câmera OK")
        print("Frame shape:", frame.shape)
    else:
        print("Câmera abriu, mas não retornou frame.")
    cap.release()
else:
    print("Câmera 0 não abriu. Use um arquivo de vídeo com --source nos exemplos.")
