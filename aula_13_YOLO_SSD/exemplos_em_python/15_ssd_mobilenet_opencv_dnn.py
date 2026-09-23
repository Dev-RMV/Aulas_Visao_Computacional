"""
COMENTÁRIO GERAL

Este exemplo pertence a uma sequência incremental sobre detecção de objetos
em tempo real, comparação entre YOLO e SSD e rastreamento com ID persistente.
A ideia é começar com situações controladas e avançar gradualmente para modelos
reais, medição de desempenho e análise crítica em aplicações embarcadas.
"""

"""
Exemplo com SSD MobileNet via OpenCV DNN.
Arquivos típicos esperados em modelos/:
- frozen_inference_graph.pb
- ssd_mobilenet_v2_coco.pbtxt
"""
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
pb = ROOT / "modelos" / "frozen_inference_graph.pb"
pbtxt = ROOT / "modelos" / "ssd_mobilenet_v2_coco.pbtxt"

if not (pb.exists() and pbtxt.exists()):
    print("Coloque os arquivos do SSD MobileNet em modelos/.")
    raise SystemExit

net = cv2.dnn.readNetFromTensorflow(str(pb), str(pbtxt))
print("SSD MobileNet carregado com OpenCV DNN.")

# DESAFIO DO ALUNO:
# Crie o blob de uma imagem e rode net.forward().
