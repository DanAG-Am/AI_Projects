'''
Nombre: Amilka Daniela Lopez Aguilar
Matrícula: A01029277
Proyecto 1
'''

# Importar librerías para manipulación de vectores y creación de gráficas
import numpy as np
import matplotlib.pyplot as plt

# Función de costo para regresión lineal
def calculaCosto(X, y, theta):
    '''
    Calcula la función de costo para regresión lineal. Recibe la matriz X, vector y, vector theta y devuelve el costo J.
    '''
    # definición de variables
    m = len(y)  # número de datos
    h = X.dot(theta)  # hipótesis, pensando en la sustitución de H por su equivalente en la ecuación de costo y con los unos en la primera columna de X

    # sustitución en la función de costo
    J = (1/(2*m)) * np.sum((h - y) ** 2)
    return J

# Función de gradiente descendiente para regresión lineal
def gradienteDescendente(X, y, theta, alpha, iteraciones):
    '''
    Calcula el gradiente descendiente para regresión lineal. Recibe la matriz X, vector y, vector theta, tasa de aprendizaje alpha y número de iteraciones.
    Devuelve el valor del vector theta final. Método batch donde usamos todos los datos:
    '''
    # definición de variables
    m = len(y)  # número de datos
    for i in range(iteraciones):
        h = X.dot(theta)  # hipótesis
        gradiente = (1/m) * X.T.dot(h - y)  # gradiente -> operaciones con matrices de la clase
        theta -= alpha * gradiente  # actualización de theta
    return theta

def graficaDatos(X, y, theta):
    '''
    Función para graficar los datos y la línea de regresión. Recibe la matriz X, vector y y vector theta.
    '''
    plt.scatter(X[:, 1], y, color='red', marker='x', label='Datos') 
    plt.plot(X[:, 1], X.dot(theta), color='blue', label='Regresión lineal') 
    plt.xlabel('X') 
    plt.ylabel('y') 
    plt.title('Regresión Lineal - Food Trucks')
    plt.legend()
    plt.show()

'''
Pase a producción, usando el txt proporcionado y en un main para evitar interferir con la llamada a funciones desde otros archivos.
'''
if __name__ == "__main__":

    #cargar datos del archivo txt
    X, y = np.loadtxt('ex1data1.txt', delimiter=',', unpack=True) # consulta para asignación de valores a vectores facil en numpy https://numpy.org/devdocs/reference/generated/numpy.loadtxt.html

    #imprimir los datos cargados - > ya validamos 
    print("Datos cargados:")
    print("X:", X) # población
    print("y:", y) # ganancias

    # agregar una columna de unos a X para la vectorización fácil
    X = np.column_stack((np.ones(len(X)), X)) # columna con el mismo numero de filas que X pero con unos

    # inicializar theta, alpha y número de iteraciones
    theta = np.zeros(X.shape[1]) # inicializamos theta con ceros

    # costo inicial
    print("Costo inicial:", calculaCosto(X, y, theta)) # verificado -> 5.8776396...

    # inicializar parámetros para el gradiente descendiente
    alpha = 0.01 # tasa de aprendizaje
    iteraciones = 1500 # número de iteraciones o pasos

    # Entrenar
    theta = gradienteDescendente(X, y, theta, alpha, iteraciones)
    print("Theta final:", theta) # verificado -> [-3.63029144  1.16636235]

    # Costo final 
    print("Costo final:", calculaCosto(X, y, theta)) # verificado -> 4.483388

    # grafica de los datos y la línea de regresión
    graficaDatos(X, y, theta)

    # prueba de código
    Prediccion1 = np.array([1, 3.5]).dot(theta) # Regresa 0.4519767868
    Prediccion2 = np.array([1, 7]).dot(theta) # Regresa 4.5342450129

    print("Resultados de las predicciones:")
    print("Predicción 1 (para X=3.5):", Prediccion1) # verificado -> 0.4519767867701767
    print("Predicción 2 (para X=7):", Prediccion2) # verificado -> 4.534245012897017

