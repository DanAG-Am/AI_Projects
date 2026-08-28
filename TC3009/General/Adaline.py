'''
Programa de perceptron Adaline o perceptron multicapa
'''
import numpy as np
import pandas as pd

# error completo de los cuatro datos debe ser menor a epsilon 
epsilon = 0.01

x = [[1,0,0],[1,0,1],[1,1,0], [1,1,1]]
y = [[1], [1], [1], [1]] # salida deseada para la compuerta AND de 2 entradas

w0 = 1.5
w1 = 0.5
w2 = 1.5
alpha = 0.1
pasos = 1000
error = False
epsilon = 0.01

while error == False:
    error_ciclo = 0
    for i in range(len(x)):
        y_pred = w0*x[i][0] + w1*x[i][1] + w2*x[i][2]
        error_ciclo += (y[i][0] - y_pred) ** 2
        w0 += alpha * (y[i][0] - y_pred) * x[i][0]
        w1 += alpha * (y[i][0] - y_pred) * x[i][1]
        w2 += alpha * (y[i][0] - y_pred) * x[i][2]
        print("Iteración", i+1, "Error:", (y[i][0] - y_pred) ** 2, "Pesos: w0 =", w0, "w1 =", w1, "w2 =", w2)
    if error_ciclo < epsilon:
        error = True
        print("Error ciclo menor a epsilon, deteniendo el entrenamiento.")
    print("Error ciclo:", error_ciclo)
    print("Pesos actuales: w0 =", w0, "w1 =", w1, "w2 =", w2)









'''
epsilon = 0.01
datos = pd.DataFrame({
    'X0': [1, 0, 0, 0],
    'X1': [1, 0, 1, 1],
    'X2': [1, 1, 0, 1],
    'Y': [1, 1, 1, 1]
})

X = datos[['X0', 'X1', 'X2']].values
Y = datos['Y'].values

# pesos iniciales 

w0 = 1.5
w1 = 0.5
w2 = 1.5
alpha = 0.1
error_inicial = epsilon

for i in range(len(X)):
    y_pred = w0*X[i][0] + w1*X[i][1] + w2*X[i][2]
    error = Y[i] - y_pred
    error_inicial += error**2
    w0 += alpha * error * X[i][0]
    w1 += alpha * error * X[i][1]
    w2 += alpha * error * X[i][2]
    print("Iteración", i+1, "Error:", error, "Pesos: w0 =", w0, "w1 =", w1, "w2 =", w2)

print("Error inicial:", error_inicial)
print("Pesos finales: w0 =", w0, "w1 =", w1, "w2 =", w2)
'''