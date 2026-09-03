'''
Nombre: Amilka Daniela Lopez Aguilar
Matrícula: A01029277
Proyecto 3: RNN con Backpropagation
'''

import numpy as np

def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    '''
    Entrena la RN con Backpropagation, recibe el tamaño de la matriz que 
    expresa las capas de entrada y ocultas, X, y y num de etiquetas 
    '''
    m = len(X) # filas o cantidad de datos en x
    alpha = 0.8
    iteraciones = 2200
    #inicialización de pesos
    W1 = randInicializacionPesos(input_layer_size, hidden_layer_size) # conecta con capa intermedia
    b1 = np.zeros((1,hidden_layer_size))
    W2 = randInicializacionPesos(hidden_layer_size, num_labels)
    b2 = np.zeros((1,num_labels))

    for i in range(iteraciones):
        #primero calcular hacia adelante, empezando por el hidden layer
        nets2 = np.dot(X,W1.T)+b1
        o2 = 1 / (1 + np.exp(-nets2))
        #capa salida
        nets3 = np.dot(o2,W2.T)+b2
        o3 = 1 / (1 + np.exp(-nets3))
        #costo
        J = -np.sum(y*np.log(o3) + (1-y)*np.log(1-o3)) / m
        #deltas o error
        delta3 = o3 - y
        delta2 = np.dot(delta3,W2)*sigmoidalGradiente(nets2) #propagar error
        #gradientes y derivadas
        dW2 = np.dot(delta3.T, o2)/m
        db2 = np.sum(delta3, axis = 0)/m # por la columna x 1 que se incluyo por las indicaciones de las dimensiones de las matrices
        dW1 = np.dot(delta2.T, X)/m
        db1 = np.sum(delta2, axis=0) / m
        #actualizar 
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
    print("Costo Final: ", J)
    return W1, b1, W2, b2

def sigmoidalGradiente(z):
    '''
    Recibe una función z y devueve 
    el gradiente para la función sigmoidal
    '''
    sigmoidal = 1 / (1 + np.exp(-z))
    return sigmoidal * (1 - sigmoidal) # g * (1-g)

def randInicializacionPesos(L_in, L_out):
    '''
    Inicializa aleatoriamente los pesos de una capa que tienen
    L_in entradas (unidades de la capa anterior, sin contar el bias) y L_out salidas (unidades de la
    capa actual). de -e a e
    '''
    e = 0.12
    tamaño_rango = 2*e
    return np.random.rand(L_out, L_in) * tamaño_rango - e # movernos en el rango de 0.24 por -0.12 y 0.12


def prediceRNYaEntrenada(X,W1,b1,W2,b2):
    '''
    Crear una red neuronal muy específica que ya fue previamente entrenada. 
    Recibe 4 matrices, W1 de 25 X 400 (porque son 400 neuronas en la capa de entrada), 
    b1 de 400 X 1, W2 de 10 X 25 (porque son 25
    neuronas en la capa intermedia) y b2 de 25 X 1, que contienen los pesos para las capas
    intermedia y de salida más el bias (b), respectivamente.
    '''
    nets2 = np.dot(X,W1.T) + b1
    o2 = 1 / (1 + np.exp(-nets2))
    nets3 = np.dot(o2, W2.T) + b2
    o3 = 1 / (1 + np.exp(-nets3))
    y = np.argmax(o3, axis=1) + 1 #seleccionamos la neurona con mayor activación, argmax sustituye el for que hacíamos para recorrer o3 -> https://numpy.org/devdocs/reference/generated/numpy.argmax.html
    return y

'''
Pase a producción, usando el txt proporcionado y
en un main para evitar interferir con la llamada a funciones desde otros archivos.
'''

if __name__ == "__main__":

    #cargar datos 
    datos = np.loadtxt("ML Algorithms/Project6/digitos.txt")
    X = datos[:, 0:400]
    y = datos[:, 400].astype(int)
    # y tiene que hacer referencia a un valor, según la expresión de un número en la matríz
    y_converted = np.zeros((len(y),10))
    for i in range(len(y)):
        label = int(y[i])
        y_converted[i, label-1]=1
    #Entrenar
    W1, b1, W2, b2 = entrenaRN(400,25,10,X,y_converted) # -> Costo Final verificado: 0.23389794438348163 vs 0.287629.
    predicciones = prediceRNYaEntrenada(X,W1,b1,W2,b2)
    #aciertos
    aciertos = np.sum(predicciones == y)
    exactitud = aciertos/len(y)*100
    print("Exactitud: ", round((exactitud),1)) # -> verificado, 97.5% vs 97.5%