import tensorflow as tf, matplotlib.pyplot as plt
m=tf.keras.models.load_model('modelo_mnist.keras'); w=m.layers[0].get_weights()[0]; print('Shape:',w.shape)
fig,axs=plt.subplots(4,4,figsize=(8,8))
for i,ax in enumerate(axs.ravel()): ax.axis('off'); ax.imshow(w[:,:,0,i],cmap='gray'); ax.set_title(f'Filtro {i}')
plt.tight_layout(); plt.savefig('16_kernels.png',dpi=150); plt.show()
