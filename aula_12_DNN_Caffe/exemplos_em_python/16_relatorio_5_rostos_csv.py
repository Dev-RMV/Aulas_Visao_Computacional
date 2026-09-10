from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 16 - Gerar relatorio de 5 rostos para validacao do Item A.
Crie um CSV manual em data/relatorio_5_rostos.csv com colunas:
imagem,genero_real,idade_real
"""
import csv, cv2
from pathlib import Path
from utils.face_utils import load_image, detect_faces_haar, crop_face, load_caffe_age_gender, predict_age_gender, ensure_dir
csv_path = Path("data/relatorio_5_rostos.csv")
if not csv_path.exists():
    csv_path.write_text("imagem,genero_real,idade_real\nface0.jpg,Masculino,(25-32)\n", encoding="utf-8")
    print("Modelo de CSV criado em", csv_path)
    print("Edite o arquivo com 5 linhas reais e rode novamente.")
    raise SystemExit
age_net, gender_net = load_caffe_age_gender()
rows_out = []
with open(csv_path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        img = load_image(Path("data/faces_teste") / row["imagem"])
        faces = detect_faces_haar(img)
        if not faces:
            rows_out.append({**row, "pred_genero":"NA", "pred_idade":"NA", "acerto_genero":False, "acerto_idade":False})
            continue
        pred = predict_age_gender(crop_face(img, faces[0]), age_net, gender_net)
        rows_out.append({**row, "pred_genero":pred.gender, "pred_idade":pred.age,
                         "acerto_genero":pred.gender == row["genero_real"], "acerto_idade":pred.age == row["idade_real"]})
ensure_dir("resultados")
out = Path("resultados/16_relatorio_5_rostos_resultado.csv")
with open(out, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
    writer.writeheader(); writer.writerows(rows_out)
for r in rows_out[:5]: print(r)
print("Relatorio salvo em", out)
