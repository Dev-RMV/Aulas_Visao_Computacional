import tensorflow as tf, matplotlib.pyplot as plt
m=tf.keras.models.load_model('modelo_mnist.keras'); (_, _),(xt,_)=tf.keras.datasets.mnist.load_data(); e=xt[0][None,...,None]/255.0
outs=[l.output for l in m.layers if isinstance(l,tf.keras.layers.Conv2D)]; vm=tf.keras.Model(m.input,outs); fm=vm.predict(e,verbose=0)[0][0]
fig,axs=plt.subplots(4,4,figsize=(8,8))
for i,ax in enumerate(axs.ravel()): ax.axis('off'); ax.imshow(fm[:,:,i],cmap='gray'); ax.set_title(f'Mapa {i}')
plt.tight_layout(); plt.savefig('17_feature_maps.png',dpi=150); plt.show()
