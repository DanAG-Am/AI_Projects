# ===================================================================================
# EJERCICIO: approximate the distribution parameters of the following distributions:
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

X1, Y1 = make_blobs(n_features=2, centers=3, random_state=43, cluster_std=0.5)
Y1=Y1.reshape(-1,1)
print("X1", X1[::10])
print("Y1", Y1[::10])
X=np.hstack((X1,Y1))
X[::10]

X[X[:,2]==1].std(axis=0)

plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")
plt.show()

X1.shape

Y1.shape

X.shape

n=np.array([[1,3],[2,3],[3,2]])

c=np.array([[1],[2],[2]])

n[:,1]==3

r=np.concatenate((n,c),axis=1)

filtro=r[:,2]==2

r[filtro]

o=r[r[:,2]==2]

o[:,0].std()

clases = [0, 1, 2]

mu_estimadas = []
sigma_estimadas = []

historial_mu = []   
historial_sigma = []

for c in clases:
    # 2 features (0, 1)
    datos = X[X[:, 2] == c][:, 0:2]

    mu = np.array([0.0, 0.0])      # mu feature 0, mu feature 1
    sigma2 = np.array([0.0, 0.0])  # sigma^2 feature 0, sigma^2 feature 1
    mu_values = []
    sigma_values = []

    for j in range(len(datos)):
        x = datos[j]
        N = j + 1

        mu = mu + (x - mu) / N
        sigma2 = sigma2 + ((x - mu) ** 2 - sigma2) / N

        mu_values.append(mu)
        sigma_values.append(sigma2 ** 0.5)   # sigma = raíz de sigma^2 !!!!!!!!!!!!!!

    mu_estimadas.append(mu)
    sigma_estimadas.append(sigma2 ** 0.5)
    historial_mu.append(np.array(mu_values))
    historial_sigma.append(np.array(sigma_values))

print("estimación secuencial:")
for c in clases:
    print(f"clase {c}: mu = {mu_estimadas[c]}, sigma = {sigma_estimadas[c]}")

print("estimación batch:")
for c in clases:
    datos = X[X[:, 2] == c][:, 0:2]
    n = len(datos)

    mu_batch = sum(datos) / n
    sigma_batch = (sum((datos - mu_batch) ** 2) / n) ** 0.5

    print(f"clase {c}: mu = {mu_batch}, sigma = {sigma_batch}, .std(axis=0) = {datos.std(axis=0)}")

for c in clases:
    plt.plot(historial_mu[c][:, 0], label=f'clase {c}, feature 0')
    plt.plot(historial_mu[c][:, 1], label=f'clase {c}, feature 1')
plt.title('convergencia de mu')
plt.xlabel('N (muestras vistas)')
plt.ylabel('mu')
plt.legend()
plt.show()