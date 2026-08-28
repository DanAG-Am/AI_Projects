import numpy as np
import pandas as pd

# error completo de los cuatro datos debe ser menor a epsilon 
epsilon = 0.001

x = [[1,0,0],[1,0,1],[1,1,0], [1,1,1]]
y = [[0], [1], [1], [0]] # salida deseada para la compuerta AND de 2 entradas
error = False


# generar 10 valores random entre 0 y 1
random_values = np.random.rand(10)
print("Random values:", random_values.round(2))

# generar 10 valores random entre -1 y 1
random_values_neg = 2 * np.random.rand(10) - 1
print("Random values (-1 to 1):", random_values_neg.round(2))

'''
Programa de Madeline BPN o Madeline de Backpropagation
'''

# inicializar variables

w13 = 0.57
w23 = 0.29
b3 = 0.83
w14 = 0.56
w24 = 0.47
b4 = 0.10
w35 = 0.14
w45 = 0.11
b5 = 0.28

alfa = 0.05
pasos = 72000

def sigmoidal(z):
    return 1 / (1 + np.exp(-z))

def derivada_sigmoidal(z):
    return sigmoidal(z) * (1 - sigmoidal(z))

# calculamos net 3 y net 4

x1 = x[0][1]
x2 = x[0][2]

while error == False:
    error_ciclo = 0
    for i in range(len(y)):
        net3 = b3 + w13 * x1 + w23 * x2
        net4 = b4 + w14 * x1 + w24 * x2

        # calculamos las salidas de las neuronas 3 y 4
        y3 = sigmoidal(net3)
        y4 = sigmoidal(net4)

        # calculamos net 5
        net5 = b5 + w35 * y3 + w45 * y4

        # calculamos la salida de la neurona 5
        y5 = sigmoidal(net5)
        print("Output y5:", y5.round(2))

        # calcular cada variable con un temp

        # actualizar pesos 
        delta5 = (y[0][0] - y5) * derivada_sigmoidal(net5)
        delta3 = delta5 * w35 * derivada_sigmoidal(net3)
        delta4 = delta5 * w45 * derivada_sigmoidal(net4)

        w13 += alfa * delta3 * x1
        w23 += alfa * delta3 * x2
        b3 += alfa * delta3

        w14 += alfa * delta4 * x1
        w24 += alfa * delta4 * x2
        b4 += alfa * delta4

        w35 += alfa * delta5 * y3
        w45 += alfa * delta5 * y4
        b5 += alfa * delta5

        error_ciclo += (y[0][0] - y5) ** 2
    if error_ciclo < epsilon:
        error = True
        print("Error ciclo menor a epsilon, deteniendo el entrenamiento.")
        break
    print("Error ciclo:", error_ciclo)
    pasos += 1
    print("Pasos:", pasos)
    print("Pesos y sesgos actualizados:")
    print("w13:", w13, "w23:", w23, "b3:", b3)
    print("w14:", w14, "w24:", w24, "b4:", b4)
    print("w35:", w35, "w45:", w45, "b5:", b5)
    print("o5", y5.round(2))
    print("-----------------------------")


