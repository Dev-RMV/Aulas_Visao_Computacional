from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 17 - Benchmark simples de latencia por imagem para o modelo Caffe.
"""
import argparse, time
from utils.face_utils import load_image, detect_faces_haar, crop_face, load_caffe_age_gender, predict_age_gender
parser = argparse.ArgumentParser(); parser.add_argument("--imagem", default="data/faces_teste/pessoa.jpg"); parser.add_argument("--repeticoes", type=int, default=30)
args = parser.parse_args()
img = load_image(args.imagem)
face = crop_face(img, detect_faces_haar(img)[0])
age_net, gender_net = load_caffe_age_gender()
for _ in range(3): predict_age_gender(face, age_net, gender_net)
t0 = time.perf_counter()
for _ in range(args.repeticoes): predict_age_gender(face, age_net, gender_net)
lat_ms = (time.perf_counter() - t0) * 1000 / args.repeticoes
print(f"Latencia media: {lat_ms:.2f} ms/imagem")
print(f"FPS teorico em uma face: {1000/lat_ms:.2f}")
