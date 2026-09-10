from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 15 - Salvar video com bounding boxes e rotulos.
"""
import argparse, cv2, time
from utils.face_utils import detect_faces_haar, crop_face, load_caffe_age_gender, predict_age_gender, draw_label, ensure_dir
parser = argparse.ArgumentParser(); parser.add_argument("--video", required=True); parser.add_argument("--saida", default="resultados/video_anotado.mp4")
args = parser.parse_args()
age_net, gender_net = load_caffe_age_gender()
cap = cv2.VideoCapture(args.video)
fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ensure_dir("resultados")
out = cv2.VideoWriter(args.saida, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w,h))
frames, t0 = 0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok: break
    for box in detect_faces_haar(frame):
        pred = predict_age_gender(crop_face(frame, box), age_net, gender_net)
        draw_label(frame, box, f"{pred.gender} | {pred.age}")
    out.write(frame); frames += 1
cap.release(); out.release()
print("Video salvo em", args.saida)
print(f"FPS medio de processamento: {frames / max(1e-6, time.perf_counter()-t0):.2f}")
