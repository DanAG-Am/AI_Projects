'''
Nombre: Amilka Daniela Lopez Aguilar
Matrícula: A01029277
Proyecto 2: Regresión Logística Vectorizada
'''

import numpy as np
import matplotlib.pyplot as plt

def leer_archivo(nombre_archivo):
    '''
    Función para leer un archivo de texto y devolver los datos en un arreglo de NumPy.
    Recibe el archivo y regresa los datos, separados x y, donde hay dos valores de x.
    '''
    datos = np.loadtxt(nombre_archivo, delimiter=',')
    x = datos[0:2, :].T
    y = datos[2, :]
    return datos

def graficar_datos(x, y):
    '''
    Función para graficar los datos de entrada.
    Recibe los datos x y y, donde x tiene dos características y y es la etiqueta.
    Utilizando un for loop
    '''
    for i in range(len(y)):
        if y[i] == 0:
            plt.scatter(x[i, 1], x[i, 2], color='red', marker='o', label='Clase 0' if i == 0 else "")
        else:
            plt.scatter(x[i, 1], x[i, 2], color='blue', marker='x', label='Clase 1' if i == 0 else "")

    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Datos de Entrenamiento')
    plt.show()

def funcion_sigmoidal(z):
    '''
    Función sigmoidal para la regresión logística.
    Recibe un valor z y devuelve el resultado de la función sigmoidal.
    '''
    return 1 / (1 + np.exp(-z))

def h_theta(x, theta_cero, theta_uno, theta_dos):
    '''
    Función de hipótesis para la regresión logística.
    Recibe los datos x y los parámetros theta, y devuelve la predicción.
    '''

    return funcion_sigmoidal(theta_cero + np.dot(x[:, 1], theta_uno) + np.dot(x[:, 2], theta_dos))

def costo(x, y, theta_cero, theta_uno, theta_dos):
    '''
    Función de costo para la regresión logística.
    Recibe los datos x, y y los parámetros theta, y devuelve el costo.
    '''

    m = len(y)
    h = h_theta(x, theta_cero, theta_uno, theta_dos)
    return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def error(x, y, theta_cero, theta_uno, theta_dos):
    '''
    Función para calcular el error de predicción.
    Recibe los datos x, y y los parámetros theta, y devuelve el error.
    '''

    h = h_theta(x, theta_cero, theta_uno, theta_dos)

    return np.mean((h >= 0.5) != y)

def gradiente_descendente(x, y, theta_cero, theta_uno, theta_dos, alpha, num_iter):
    '''
    Función para realizar el gradiente descendente.
    Recibe los datos x, y, los parámetros theta, la tasa de aprendizaje alpha y
    el número de iteraciones.
    Devuelve los parámetros theta actualizados y el historial de costos.
    '''

    m = len(y)
    costos = []

    for i in range(num_iter):

        h = h_theta(x, theta_cero, theta_uno, theta_dos)

        gradiente_cero = (1/m) * np.sum(h - y)
        gradiente_uno = (1/m) * np.sum((h - y) * x[:, 1])
        gradiente_dos = (1/m) * np.sum((h - y) * x[:, 2])

        theta_cero -= alpha * gradiente_cero
        theta_uno -= alpha * gradiente_uno
        theta_dos -= alpha * gradiente_dos

        costos.append(costo(x, y, theta_cero, theta_uno, theta_dos))

    return theta_cero, theta_uno, theta_dos, costos

def prediccion(x, theta_cero, theta_uno, theta_dos):
    '''
    Función para realizar predicciones con los parámetros theta.
    Recibe los datos x y los parámetros theta, y devuelve las predicciones.
    '''
    h = h_theta(x, theta_cero, theta_uno, theta_dos)
    predicciones = []
    for i in range(len(h)):
        if h[i] >= 0.5:
            predicciones.append(1)
        else:
            predicciones.append(0)
    return np.array(predicciones)

def accuracy(y_true, y_pred):
    '''
    Función para calcular la exactitud de las predicciones.
    Recibe los valores verdaderos y las predicciones, y devuelve la exactitud.
    '''
    return np.mean(y_true == y_pred) * 100

def recta_limite(x, y, theta_cero, theta_uno, theta_dos):
    '''
    Función para graficar la recta de límite, punto en medio de la sigmoidal
    Usando x1 y x2, donde x1 es la primera característica y x2 es la segunda característica.
    Recibe los datos x, y y los parámetros theta, y grafica la recta de límite.
    theta cero + theta_uno * x1 + theta_dos * x2 = 0
    '''
    for i in range(len(y)):
        if y[i] == 0:
            plt.scatter(x[i, 1], x[i, 2], color='red', marker='o', label='Clase 0' if i == 0 else "")
        else:
            plt.scatter(x[i, 1], x[i, 2], color='blue', marker='x', label='Clase 1' if i == 0 else "")
            x1 = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 100)
            x2 = -(theta_cero + theta_uno * x1) / theta_dos
            plt.plot(x1, x2, color='green', label='Recta de límite')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Datos de Entrenamiento con Recta de Límite')
    plt.show()

'''
Pase a producción, usando el csv proporcionado y
en un main para evitar interferir con la llamada a funciones desde otros archivos.
'''

if __name__ == "__main__":

    datos = leer_archivo("TC3009/ML Algorithms/Project2/RegLog.csv")

    x = datos[:, 0:2]
    y = datos[:, 2]

    x = np.column_stack((np.ones(len(x)), x))

    graficar_datos(x, y)

    theta_cero = 0.0
    theta_uno = 0.0
    theta_dos = 0.0

    alpha = 0.1
    num_iter = 4000

    theta_cero, theta_uno, theta_dos, costos = gradiente_descendente(x, y, theta_cero, theta_uno, theta_dos, alpha, num_iter)

    costo_final = costo(x, y, theta_cero, theta_uno, theta_dos)

    print("Costo final:", costo_final)

    predicciones = prediccion(x, theta_cero, theta_uno, theta_dos)

    exactitud = accuracy(y, predicciones)

    print("Exactitud del modelo:", exactitud)

    recta_limite(x, y, theta_cero, theta_uno, theta_dos)