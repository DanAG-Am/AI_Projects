'''
A01029277 Amilka Daniela Lopez Aguilar
Estimación secuencial y batch de parámetros (mu y sigma) para un conjunto de datos generado con make_blobs.
Se comparan las estimaciones secuenciales con las estimaciones batch y se visualiza la convergencia de mu y sigma.
'''
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X1, Y1 = make_blobs(n_features=2, centers=3, random_state=43, cluster_std=0.5)
X = np.hstack((X1, Y1.reshape(-1, 1)))

plt.scatter(X1[:, 0], X1[:, 1], c=Y1, s=25, edgecolor="k")
plt.show()

clases = [0, 1, 2]
mu_estimadas = []
sigma_estimadas = []
historial_mu = []
historial_sigma = []

for c in clases:
    datos = X[X[:, 2] == c][:, 0:2]
    mu = np.zeros(2)
    sigma2 = np.zeros(2)
    mu_values = []
    sigma_values = []
    for N, x in enumerate(datos, start=1):
        mu = mu + (x - mu) / N
        sigma2 = sigma2 + ((x - mu) ** 2 - sigma2) / N
        mu_values.append(mu)
        sigma_values.append(sigma2 ** 0.5)
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
    print(f"clase {c}: mu = {datos.mean(axis=0)}, sigma = {datos.std(axis=0)}")

for c in clases:
    for f in range(2):
        linea, = plt.plot(historial_mu[c][:, f], label=f'mu clase {c}, feature {f}')
        plt.plot(historial_sigma[c][:, f], '--', color=linea.get_color())
plt.title('convergencia de mu (sólida) y sigma (punteada)')
plt.xlabel('N (muestras vistas)')
plt.ylabel('valor')
plt.show()

for c in clases:
    datos = X[X[:, 2] == c][:, 0:2]
    plt.plot(np.abs(historial_mu[c] - datos.mean(axis=0)).sum(axis=1), label=f'mu clase {c}')
    plt.plot(np.abs(historial_sigma[c] - datos.std(axis=0)).sum(axis=1), '--', label=f'sigma clase {c}')
plt.title('curva de error vs batch')
plt.xlabel('N (muestras vistas)')
plt.ylabel('error absoluto')
plt.legend()
plt.show()

x = np.linspace(X1[:, 1].min() - 1, X1[:, 1].max() + 1, 300)
for c in clases:
    plt.hist(X[X[:, 2] == c][:, 1], bins=15, density=True, alpha=0.3)
    mu, s = mu_estimadas[c][1], sigma_estimadas[c][1]
    plt.plot(x, np.exp(-(x - mu) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi)), label=f'clase {c}')
plt.title('PDF con mu y sigma estimados (feature 1)')
plt.legend()
plt.show()