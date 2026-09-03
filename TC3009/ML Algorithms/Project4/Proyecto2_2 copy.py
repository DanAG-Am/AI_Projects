'''
Nombre: Amilka Daniela Lopez Aguilar
Matrícula: A01029277
Proyecto 2.2: One vs All
'''
# importar librerías matemáticas y de graficación
import numpy as np
import matplotlib.pyplot as plt

'''
Hacer un programa en Python que implemente One vs All regresión logística para identificar puntos en
2D de 3 clases.
El programa recibe como entrada un conjunto de coordenadas (x1,x2,y), y regresa los
valores de Theta (q0 … qn) de cada uno de los clasificadores necesarios para clasificar los datos
correctamente.
Las funciones deben ser calculadas por el algoritmo de regresión logística usando
regularización.
Debemos entrenar regresor 1,2,3, correr en paralelo y ver cual da probabilidad alta
Dividir datos en 80/20, hay que mezclar los datos
'''
def graficaDatos(X,y,Theta):
    '''
    Grafico considerando que ahora tenemos 3 entradas
    '''
    colores = ['r', 'b', 'g']
    marcadores = ['x', 'o', '^']

    for i in range(len(y)):
        clase = int(y[i])
        plt.scatter(X[i,0], X[i,1], c=colores[clase], marker=marcadores[clase])

    x1 = np.linspace(-1, 1.5, 100)
    x2 = np.linspace(-1, 1.5, 100)

    frontera_x1 = []
    frontera_x2 = []

    if Theta is not None:
        for i in range(len(x1)):
            for j in range(len(x2)):
                punto = np.array([[x1[i], x2[j]]])
                X_mapeado = mapeoCaracterísticas(punto)
                resultado = np.dot(X_mapeado, Theta)

                if abs(resultado[0]) < 0.05:
                    frontera_x1.append(x1[i])
                    frontera_x2.append(x2[j])

        plt.scatter(frontera_x1, frontera_x2, c='k', s=5)

    plt.show()

def sigmoidal(z):
    '''
    Recibe z y regresa su sigmoidal
    '''
    return 1 / (1 + np.exp(-z))

def mapeoCaracterísticas(X):
    '''
    Recibe la matriz X original de entradas y regresa una matriz
    en la que cada renglón queda mapeado a todos los términos polinomiales de x1 y x2 hasta la sexta
    potencia. Es decir, la función deberá regresar una matriz en donde cada renglón es:
    [1, x1, x2, x12, x1x2, x22, x13, …, x1x25, x26]T
    '''
    m = X.shape[0]
    X_mapeado = np.ones((m,1))

    for i in range(1,7):
        for j in range(i+1):
            nuevo_termino = (X[:,0]**(i-j)) * (X[:,1]**j)
            X_mapeado = np.hstack((X_mapeado, nuevo_termino.reshape(m,1)))

    return X_mapeado

def funcionCostoReg(Theta, X, y, Lambda):
    '''
    Recibe el vector de entrada X, el de salida y, y un vector
    theta y el valor de lambda para la regularización. Debe regresar la función de costo J y el
    gradiente grad. Es decir, debe regresar las variables [J,grad].
    '''
    m = len(y)
    X_mapeado = mapeoCaracterísticas(X)
    h = sigmoidal(np.dot(X_mapeado, Theta))

    h = np.clip(h, 1e-15, 1 - 1e-15)

    J = (-1/m) * (
        np.dot(y, np.log(h)) +
        np.dot((1-y), np.log(1-h))
    ) + (Lambda/(2*m)) * np.sum(Theta[1:]**2)

    grad = (1/m) * np.dot(X_mapeado.T, (h-y))
    grad[1:] = grad[1:] + (Lambda/m) * Theta[1:]

    return J, grad

def oneVsAll(X, y, K, Lambda, iteraciones, alpha):
    '''
    Implementa la clasificación one-vs-all mediante el entrenamiento de
    múltiples clasificadores con regresión logística, uno para cada una de las K clases del conjunto
    de datos. En el caso de los puntos de los ejemplos K = 3.
    La función debe regresar todos los parámetros de los clasificadores en una matriz Θ ∈
    Rkx(n+1), donde cada columna de Q corresponde a los parámetros aprendidos de un
    clasificador. Se puede hacer esto con loop de 1 a K, entrenando cada clasificador independientemente.
    El vector y tiene etiquetas {0,…,2}, sin embargo, cuando entrena el clasificador j-ésimo lo que
    desea realmente es tener un vector donde todas las etiquetas diferentes de j son 0 y todas
    las de j son 1. Tiene que formar ese vector antes de entrenar el clasificador j.
    Use 100 iteraciones y lambda 0.1.
    '''
    X_mapeado = mapeoCaracterísticas(X)
    all_theta = np.zeros((K, X_mapeado.shape[1]))

    for clase in range(K):
        Theta = np.zeros(X_mapeado.shape[1])
        y_clase = (y == clase).astype(int)

        for i in range(iteraciones):
            J, grad = funcionCostoReg(Theta, X, y_clase, Lambda)
            Theta = Theta - alpha * grad

        all_theta[clase] = Theta

    return all_theta

def prediceOneVsAll(all_theta, X):
    '''
    Recibe un vector all_theta (para los K clasificadores) y un
    vector X para varios puntos. Regresa el vector p de predicción sobre su clase el cual se
    determina con el clasificador que regrese una mayor probabilidad. Si se prueba con el mismo
    vector X y y de entrenamiento debe tener una exactitud arriba del 90% (es decir, clasifica más
    del 90% de los ejemplos en forma correcta). Observe que K = size(all_theta,1).
    '''
    X_mapeado = mapeoCaracterísticas(X)
    h = sigmoidal(np.dot(X_mapeado, all_theta.T))
    predicciones = np.argmax(h, axis=1)

    return predicciones

'''
Pase a producción, usando el txt proporcionado y
en un main para evitar interferir con la llamada a funciones desde otros archivos.
'''

if __name__ == "__main__":

    # inicializar variables
    iteraciones = 100
    Lambda = 0.1
    alpha = 0.5
    K = 3

    # Cargar los datos desde el txt
    datos = np.loadtxt("TC3009/ML Algorithms/Project4/Datos3Clases2D.csv", delimiter=",", unpack=True)

    # extraer datos para primer grafico 
    x1 = datos[0,:]
    x2 = datos[1,:]
    y = datos[2,:].astype(int)
    X = np.vstack((x1, x2)).T

    # Grafico inicial 
    graficaDatos(X, y, None)

    Theta = np.zeros(mapeoCaracterísticas(X).shape[1])

    # calculamos la función de costo y el gradiente
    y_clase = (y == 0).astype(int)
    J, gradiente = funcionCostoReg(Theta, X, y_clase, Lambda)

    print("Costo inicial:", J)
    print("Gradiente inicial:", gradiente)

    aprende = oneVsAll(X, y, K, Lambda, iteraciones, alpha)

    print("Theta de los clasificadores:")
    print(aprende)

    # hacer predicciones con el modelo entrenado
    predicciones = prediceOneVsAll(aprende, X)
    print("Predicciones:", predicciones)

    # Grafico inicial 
    graficaDatos(X, y, None)

    exactitud = np.mean(predicciones == y) * 100
    print("Exactitud del modelo con Lambda = 0.1:", exactitud, "%")

    # pruebas 2,2 -> 3
    # pruebas 6,5 -> 1
    # pruebas 10,0 -> 2
    # pruebas 5,1 -> ?

    puntos_prueba = np.array([
        [2, 2],
        [6, 5],
        [10, 0],
        [5, 1]
    ])

    predicciones_prueba = prediceOneVsAll(aprende, puntos_prueba)

    print("Predicciones para puntos nuevos:")

    for i in range(len(puntos_prueba)):
        print(
            puntos_prueba[i],
            "-> clase",
            predicciones_prueba[i]
        )
