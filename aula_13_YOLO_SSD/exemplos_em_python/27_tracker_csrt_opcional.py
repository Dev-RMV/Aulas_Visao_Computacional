"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
O OpenCV também possui rastreadores como CSRT, KCF e MIL.
Eles acompanham uma região depois de inicializada, mas não substituem um detector
em cenas com novos objetos entrando continuamente.
"""
import cv2

print("Exemplo conceitual de inicialização CSRT:")
print("tracker = cv2.legacy.TrackerCSRT_create()  # em algumas versões")
print("tracker.init(frame, bbox)")
print("ok, bbox = tracker.update(frame)")

# DESAFIO DO ALUNO:
# Verifique sua versão do OpenCV e descubra se o CSRT está em cv2 ou cv2.legacy.
