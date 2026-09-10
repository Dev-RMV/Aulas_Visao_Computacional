from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 23 - Visualizar um batch do UTKFace com labels.
"""
import tensorflow as tf, pandas as pd, matplotlib.pyplot as plt
IMG_SIZE=(160,160)
def load_sample(path, label):
    img=tf.io.read_file(path); img=tf.image.decode_jpeg(img, channels=3); img=tf.image.resize(img, IMG_SIZE)
    return img/255.0, tf.cast(label, tf.float32)
df=pd.read_csv("resultados/21_train.csv")
ds=tf.data.Dataset.from_tensor_slices((df["path"].values, df["gender"].values)).map(load_sample).batch(9)
imgs, labels = next(iter(ds))
plt.figure(figsize=(7,7))
for i in range(min(9, len(imgs))):
    plt.subplot(3,3,i+1); plt.imshow(imgs[i]); plt.axis('off'); plt.title("F" if labels[i].numpy()==1 else "M")
plt.tight_layout(); plt.savefig("resultados/23_batch_utkface.png", dpi=150); print("Salvo em resultados/23_batch_utkface.png")
