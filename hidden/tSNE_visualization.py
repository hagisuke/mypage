import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Xs = np.load("MCD_s.npy")
Xt = np.load("MCD_t.npy")
# if you want random sampleing then uncoment below 2 lines
# Xs = Xs[np.random.choice(Xs.shape[0], 10000, replace=False), :]
# Xt = Xt[np.random.choice(Xt.shape[0], 10000, replace=False), :]
X = np.concatenate((Xs,Xt), axis=0)
print(Xs.shape, Xt.shape)

# if X.shape=(len(X), 2) then you don't need tsne embedding
tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
X_embedded = tsne.fit_transform(X)
print(X_embedded[:len(Xs)].shape, X_embedded[len(Xs):].shape)

plt.scatter(X_embedded[:len(Xs),0], X_embedded[:len(Xs),1], s=0.01, c="blue")
plt.scatter(X_embedded[len(Xs):,0], X_embedded[len(Xs):,1], s=0.01, c="red")
plt.axis([-80, 80, -80, 80])
plt.axes().set_aspect('equal')
plt.show()
