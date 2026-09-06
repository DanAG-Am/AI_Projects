'''
Nombre: Amilka Daniela Lopez Aguilar
Matrícula: A01029277
Proyecto 4: K-means
'''

import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

def findClosestCentroids(X, initial_centroids):
    '''
    Recibe un conjunto de ejemplos X y los centroides
    iniciales initial_centroids. La función regresa un vector idx que contiene los índices de los
    clusters más cercanos a cada ejemplo.
    '''
    idx = np.zeros(len(X))
    for i in range(len(X)):
        distancias = np.sum((initial_centroids - X[i]) ** 2, axis=1)
        idx[i] = np.argmin(distancias) # funciona como argmax, pero ahora "gana" la menor distancia
    return idx

def computeCentroids(X,idx,K):
    '''
    Recibe el conjunto de ejemplos X, los índices a los clusters más
    cercanos idx y el número de clusters K. La función calcula las madias basadas en los centroides
    más cercanos encontrados con la función del punto 1. La función regresa un arreglo que
    contiene las coordenadas de los nuevos centroides después de haber calculado la media de los
    puntos más cercanos a los mismos.
    '''
    centroids = []
    for i in range(K):
        puntos = X[idx == i]
        centro = np.mean(puntos, axis=0) # axis = 0 para promediar sobre la coordenada correcta (x con x, y con y)
        centroids.append(centro)
    return np.array(centroids)

def runkMeans(X, initial_centroids, max_iters, true):
    '''
    Recibe el conjunto de ejemplos X, los
    centroides iniciales initial_centroids, el máximo número de iteraciones en el proceso max_iter y
    un valor booleano (OPCIONAL) que si es TRUE hace que se vayan dibujando las posiciones de los
    centroides en cada iteración. 
    '''
    centroids = initial_centroids
    K = centroids.shape[0]
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)
        if true:
            colores = ["red", "blue", "green"]
            plt.figure()
            for j in range(K):
                puntos = X[idx == j]
                plt.scatter(puntos[:, 0], puntos[:, 1],
                            color=colores[j])
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        color="black", marker="x", s=100)
            plt.show()
    return centroids, idx
                
def kMeansInitCentroids(X, K): 
    '''
    Recibe el conjunto de ejemplos X y el número de clusters K. Regresa
    K centroides seleccionados en forma aleatoria del conjunto de datos
    '''
    # se investiga como permutar y mezclar para hacerlo de forma efectiva https://note.nkmk.me/en/python-random-shuffle/
    return X[random.sample(range(len(X)), K)]

'''
Pase a producción, usando el txt proporcionado y
en un main para evitar interferir con la llamada a funciones desde otros archivos.
'''

if __name__ == "__main__":
    '''
    Parte 1
    '''
    # Cargar los datos desde el txt
    project_dir = Path(__file__).parent
    X = np.loadtxt(project_dir / "ex7data2.txt")
    # Inicializar parametros 
    K = 3
    initial_centroids = np.array([[3, 3],[6, 2],[8, 5]])
    # Comprobar primeros 3 ejemplos
    idx = findClosestCentroids(X, initial_centroids)
    print("Clusters de los primeros 3 ejemplos:")
    print(idx[:3]) # -> verificado [0. 2. 1.] = idx(1)=1, idx(2)=3 y idx(3)=2.
    # Comprobar nuevos centroides
    centroids = computeCentroids(X, idx, K)
    print("Nuevos centroides:")
    print(centroids) # -> verificado [2.42830111 3.15792418][5.81350331 2.63365645][7.11938687 3.6166844 ] vs (2.428301, 3.157924), (5.813503, 2.633656), (7.119387, 3.616684)
    # Centroides después de 10 iteraciones 
    centroids, idx = runkMeans(X,initial_centroids,10,False)
    # Nuevos centroides
    nuevos_centroides = kMeansInitCentroids(X, K)
    print("Centroides iniciales aleatorios:")
    print(nuevos_centroides)

    '''
    Parte 2: usar K-Means para comprimir una imagen
    '''
    archivo_imagen = project_dir / "bird_small.txt"
    filas, columnas, canales = map(int, np.loadtxt(archivo_imagen, max_rows=1)) # mapeo de valores a su representacion
    pixeles = np.loadtxt(archivo_imagen, skiprows=1)
    # La primera fila contiene las dimensiones
    print("Dimensiones:", filas, columnas, canales)
    # El resto de los datos son los valores RGB
    print("Cantidad de valores RGB:", len(pixeles))
    A = pixeles.reshape(filas, columnas, canales, order="F").astype(np.uint8)# recorrer columna por columna y usar uint8 para evitar warning de matplotlib https://stackoverflow.com/questions/43129422/plotting-non-uint8-images-with-matplotlib
    # Img original para asegurar que se está construyendo bien
    plt.figure(figsize=(6, 6))
    plt.imshow(A)
    plt.title("Imagen original")
    plt.show()
    # matriz m x 3
    m = filas * columnas
    X = A.reshape(m, 3)
    print("Dimensiones de X:", X.shape)
    # Aplicar K-means con 16 valores
    K = 16
    initial_centroids = kMeansInitCentroids(X, K)
    print("Centroides iniciales:")
    print(initial_centroids)
    centroides, asignaciones = runkMeans(X,initial_centroids,100,False)
    print("Centroides encontrados:")
    print(centroides)
    # Cada píxel se reemplaza por el color de su centroide
    X_comprimida = centroides[asignaciones.astype(int)]
    X_comprimida = X_comprimida.astype(np.uint8)
    imagen_comprimida = X_comprimida.reshape(filas,columnas,canales)
    # visualizar img original y comprimida en subplots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.title("Imagen original")
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_comprimida)
    plt.title("Imagen comprimida - 16 colores")
    plt.show()
