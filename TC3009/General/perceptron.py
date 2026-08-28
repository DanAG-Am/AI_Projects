'''
Programa de perceptron
'''
import numpy as np
import pandas as pd

# OR perceptron

datos = pd.DataFrame({
    'X0': [1, 0, 0, 0],
    'X1': [1, 0, 1, 1],
    'X2': [1, 1, 0, 1],
    'Y': [0, 0, 0, 1]
})

# pesos iniciales 

w0 = 1.5
w1 = 0.5
w2 = 1.5
alpha = 0.1
pasos = 1000
error = False
epsilon = 0.01

while error == False:
    error_ciclo = 0
    for i in range(len(datos)):
        x0 = datos.loc[i, 'X0']
        x1 = datos.loc[i, 'X1']
        x2 = datos.loc[i, 'X2']
        y = datos.loc[i, 'Y']

        y_pred = w0 * x0 + w1 * x1 + w2 * x2
        error_ciclo += (y - y_pred) ** 2
        w0 += alpha * (y - y_pred) * x0
        w1 += alpha * (y - y_pred) * x1
        w2 += alpha * (y - y_pred) * x2
        print("Iteración", i+1, "Error:", (y - y_pred) ** 2, "Pesos: w0 =", w0, "w1 =", w1, "w2 =", w2)
    if error_ciclo < epsilon:
        error = True
        print("Error ciclo menor a epsilon, deteniendo el entrenamiento.")
    print("Error ciclo:", error_ciclo)
    print("Pesos actuales: w0 =", w0, "w1 =", w1, "w2 =", w2)