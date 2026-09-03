'''
Nombre: Amilka Daniela Lopez Aguilar
Matrícula: A01029277
Proyecto 2.2: One vs All
'''
# importar librerías matemáticas y de graficación
import numpy as np
import random as rd
import matplotlib.pyplot as plt

'''
Hacer un programa en Python que implemente One vs All regresión logística para identificar puntos en
2D de 3 clases.
El programa recibe como entrada un conjunto de coordenadas (x1,x2,y), y regresa los
valores de Theta (q0 … qn) de cada uno de los clasificadores necesarios para clasificar los datos
correctamente.
Debemos entrenar regresor 1,2,3, correr en paralelo y ver cual da probabilidad alta
Dividir datos en 80/20, hay que mezclar los datos
'''
def graficaDatos(X,y):
    '''
    Grafico considerando que ahora tenemos 3 clases. Quitando lo de frontera por ahora.
    '''
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i,0], X[i,1], c='r', marker='x')
        elif y[i] == 2:
            plt.scatter(X[i,0], X[i,1], c='g', marker='s')
        elif y[i] == 0:
            plt.scatter(X[i,0], X[i,1], c='b', marker='o')
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
    # Agregar la columna de unos
    m = X.shape[0]
    X_mapeado = np.ones((m,1)) # -> longitud de x y relleno de 1s
    for i in range(1,7): # grado 6 
        for j in range(i+1): # para aumentar el grado cada vez que pase el loop de arriba
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
    h = sigmoidal(np.dot(mapeoCaracterísticas(X), Theta))
    J = (-1/m) * (np.dot(y, np.log(h)) + np.dot((1-y), np.log(1-h))) + (Lambda/(2*m)) * np.sum(Theta[1:]**2)
    grad = (1/m) * np.dot(mapeoCaracterísticas(X).T, (h-y))
    grad[1:] = grad[1:] + (Lambda/m) * Theta[1:]
    return J, grad

def oneVsAll(X, y, K, Lambda, iteraciones, alpha, Theta):
    '''
    La función debe regresar todos los parámetros de los clasificadores en una matriz Θ ∈
    Rkx(n+1), donde cada columna de Q corresponde a los parámetros aprendidos de un
    clasificador.
    '''
    X_mapeado = mapeoCaracterísticas(X)
    all_theta = np.zeros((K, X_mapeado.shape[1]))
    for clase in range(K):
        Theta = np.zeros(X_mapeado.shape[1])
        y_clase = (y == clase) # 0, 1
        for i in range(iteraciones):
            J, grad = funcionCostoReg(Theta, X, y_clase, Lambda)
            Theta = Theta - alpha * grad
        all_theta[clase] = Theta
    return all_theta
  
def prediceOneVsAll(all_theta, X):
    '''
    Recibe un vector all_theta (para los K clasificadores) y un
    vector X para varios puntos. Regresa el vector p de predicción sobre su clase el cual se
    determina con el clasificador que regrese una mayor probabilidad. 
    '''
    X_mapeado = mapeoCaracterísticas(X)
    h = sigmoidal(np.dot(X_mapeado, all_theta.T))
    predicciones = np.argmax(h, axis=1) # Tomamos el máximo para definir qué clase "gana" en la predicción
    return predicciones

'''
Pase a producción, usando el txt proporcionado y
en un main para evitar interferir con la llamada a funciones desde otros archivos.
'''

if __name__ == "__main__":

    # inicializar variables
    Theta = np.zeros(28)  # Inicializar el vector de parámetros theta con ceros (28 características después del mapeo)
    iteraciones = 2000  # Número de iteraciones para el gradiente descendente
    Lambda = 0.1  # Valor de lambda para la regularización
    alpha = 0.1  # Tasa de aprendizaje para el gradiente descendente
    K = 3

    # Cargar los datos desde el txt
    datos = np.loadtxt("TC3009/ML Algorithms/Project4/Datos3Clases2D.csv", delimiter=",", unpack=True)

    # extraer datos para primer grafico 
    x1 = datos[0,:]
    x2 = datos[1,:]
    y = datos[2,:]
    X = np.vstack((x1, x2)).T  # Combinar x1 y x2 en una matriz de características X, transpuesta para su lectura

    # shuffle de los datos
    indices = list(range(len(y)))
    rd.shuffle(indices)
    X = X[indices]
    y = y[indices]
    print("Datos mezclados:", len(y))

    # Grafico inicial 
    graficaDatos(X, y)

    # 80/20 
    corte = int(len(y) * 0.80)

    entrenamiento = X[:corte] # antes del corte, el *80
    y_entrenamiento = y[:corte]
    prueba = X[corte:] # despues del corte, el *20
    y_prueba = y[corte:]

    # calculamos la función de costo y el gradiente
    J, gradiente = funcionCostoReg(Theta, entrenamiento, y_entrenamiento, Lambda)
    print("Costo inicial:", J)
    print("Gradiente inicial:", gradiente)

    aprende = oneVsAll(entrenamiento, y_entrenamiento, K, Lambda, iteraciones, alpha, Theta)
    print("Thetas obtenidas:", aprende)

    # hacer predicciones con el modelo entrenado
    predicciones = prediceOneVsAll(aprende, prueba)
    print("Predicciones:", predicciones)

    exactitud = np.mean(predicciones == y_prueba) * 100
    print("Exactitud del modelo con Lambda = 0.1:", exactitud, "%")

    # pruebas 2,2 -> 0
    # pruebas 6,5 -> 1
    # pruebas 10,0 -> 2
    # pruebas 5,1 -> ?

    puntos_prueba = np.array([[2, 2],[6, 5],[10, 0],[5, 1]])
    predicciones_prueba = prediceOneVsAll(aprende, puntos_prueba)
    print("Predicciones para puntos nuevos:")
    for i in range(len(puntos_prueba)):
        print(puntos_prueba[i],"-> clase",predicciones_prueba[i])

    '''
    verificado: [2 2] -> clase 0
                [6 5] -> clase 1
                [10  0] -> clase 2
                [5 1] -> clase 2
    '''