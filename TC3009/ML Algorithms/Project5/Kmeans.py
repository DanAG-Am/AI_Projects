'''
Nombre: Amilka Daniela Lopez Aguilar
Matrícula: A01029277
Proyecto 3: K-means
'''

import numpy as np
import matplotlib.pyplot as plt


# Cargar los datos desde el txt
datos = np.loadtxt("ML Algorithms/Project5/ex7data2.txt", unpack=True)

x = datos[0, :]
y = datos[1, :]
n_clusters = 3
intentos = 100
max_iteraciones = 100
mejor_error = 1000
mejores_centroides = []
mejores_grupos = []

for intento in range(intentos):
    # Seleccionar centroides iniciales aleatorios sin repetir
    indices_centroides = []
    while len(indices_centroides) < n_clusters:
        indice = np.random.randint(0, len(x))
        if indice not in indices_centroides:
            indices_centroides.append(indice)

    centroides = []
    for indice in indices_centroides:
        centroides.append([x[indice], y[indice]])

    # Ejecutar una vez K-means hasta converger
    for iteracion in range(max_iteraciones):
        grupos = []
        for indice in range(n_clusters):
            grupos.append([])

        for indice_punto in range(len(x)):
            distancia_minima = float("inf")
            grupo_cercano = 0

            for indice_centroide in range(n_clusters):
                diferencia_x = x[indice_punto] - centroides[indice_centroide][0]
                diferencia_y = y[indice_punto] - centroides[indice_centroide][1]
                distancia = (diferencia_x ** 2) + (diferencia_y ** 2)

                if distancia < distancia_minima:
                    distancia_minima = distancia
                    grupo_cercano = indice_centroide

            grupos[grupo_cercano].append(indice_punto)

        nuevos_centroides = []
        for indice_centroide in range(n_clusters):
            if len(grupos[indice_centroide]) == 0:
                nuevos_centroides.append(centroides[indice_centroide])
            else:
                suma_x = 0
                suma_y = 0
                for indice_punto in grupos[indice_centroide]:
                    suma_x += x[indice_punto]
                    suma_y += y[indice_punto]

                nuevos_centroides.append([
                    suma_x / len(grupos[indice_centroide]),
                    suma_y / len(grupos[indice_centroide]),
                ])

        centroides_iguales = True
        for indice_centroide in range(n_clusters):
            if (centroides[indice_centroide][0] != nuevos_centroides[indice_centroide][0]
                    or centroides[indice_centroide][1] != nuevos_centroides[indice_centroide][1]):
                centroides_iguales = False

        centroides = nuevos_centroides
        if centroides_iguales:
            break

    # Medir el error cuadratico total de este intento
    error = 0
    for indice_grupo in range(n_clusters):
        for indice_punto in grupos[indice_grupo]:
            diferencia_x = x[indice_punto] - centroides[indice_grupo][0]
            diferencia_y = y[indice_punto] - centroides[indice_grupo][1]
            error += (diferencia_x ** 2) + (diferencia_y ** 2)

    if error < mejor_error:
        mejor_error = error
        mejores_centroides = []
        for centroide in centroides:
            mejores_centroides.append([centroide[0], centroide[1]])

        mejores_grupos = []
        for grupo in grupos:
            mejores_grupos.append([])
            for indice_punto in grupo:
                mejores_grupos[-1].append(indice_punto)

centroides = mejores_centroides
grupos = mejores_grupos

print("Centroides finales:")
for centroide in centroides:
    print(centroide)
print("Mejor error:", mejor_error)

colores = ["red", "blue", "green"]
for indice_grupo in range(n_clusters):
    puntos_x = []
    puntos_y = []
    for indice_punto in grupos[indice_grupo]:
        puntos_x.append(x[indice_punto])
        puntos_y.append(y[indice_punto])
    plt.scatter(
        puntos_x,
        puntos_y,
        color=colores[indice_grupo],
        label="Cluster " + str(indice_grupo + 1),
    )

centroides_x = []
centroides_y = []
for centroide in centroides:
    centroides_x.append(centroide[0])
    centroides_y.append(centroide[1])

plt.scatter(centroides_x, centroides_y, color="black", marker="x", s=100,
            label="Centroides")
plt.legend()
plt.show()




