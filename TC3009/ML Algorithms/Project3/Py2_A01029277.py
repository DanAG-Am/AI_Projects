'''
Nombre: Amilka Daniela Lopez Aguilar
Matrícula: A01029277
Proyecto 2.1: Regresión Logística Vectorizada y con Regularización
'''
# importar librerías matemáticas y de graficación
import numpy as np
import matplotlib.pyplot as plt

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

def graficaDatos(X,y,Theta):
    '''
    Recibe el vector de entradas X, salidas y, y un vector theta. Grafica los
    datos en 2D. Los ejes son las dos pruebas y pone una X si el chip se aceptó y una O si no fue
    aceptado. También grafica la función cuyos parámetros son los del vector theta dado (con las 28
    características que regresa la función 2.
    '''
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i,0], X[i,1], c='r', marker='x')
        else:
            plt.scatter(X[i,0], X[i,1], c='b', marker='o')
    # grafica la función cuyos parámetros son los del vector theta dado (con las 28 características que regresa la función 2.
    # Puntos para la frontera
    x1 = np.linspace(-1, 1.5, 100)
    x2 = np.linspace(-1, 1.5, 100)
    # Guardar puntos de la frontera
    frontera_x1 = []
    frontera_x2 = []
    for i in range(len(x1)):
        for j in range(len(x2)):
            # Punto actual
            punto = np.array([[x1[i], x2[j]]])
            # Obtener las 28 características
            X_mapeado = mapeoCaracterísticas(punto)
            # Calcular Theta * X
            resultado = np.dot(X_mapeado, Theta)
            # Frontera cuando el resultado es cercano a 0
            if abs(resultado[0]) < 0.05:
                frontera_x1.append(x1[i])
                frontera_x2.append(x2[j])
    # Graficar frontera
    plt.scatter(frontera_x1, frontera_x2, c='g', s=5) # tamaño distinto para diferenciar junto con el color
    plt.show()

def sigmoidal(z):
    '''
    Recibe z y regresa su sigmoidal
    '''
    return 1 / (1 + np.exp(-z))

def funcionCostoReg(Theta,X,y,Lambda):  
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

def aprende(Theta,X,y,iteraciones):
    '''
    Recibe el vector de entrada X (ya aumentado), el de salida y, un
    vector theta inicial (el cual puede ser el vector 0) y el número de iteraciones que se correrá el
    gradiente descendente. Debe regresar el vector de parámetros theta, encontrado por gradiente
    descendente.
    '''
    global alpha
    for i in range(iteraciones):
        J, gradiente = funcionCostoReg(Theta, X, y, Lambda)
        Theta = Theta - alpha * gradiente
    return Theta

def predice(Theta,X):
    '''
    Recibe un vector theta y un vector X para varios chips. Regresa el vector p de
    predicción sobre su aceptación utilizando un valor de umbral natural de 0.5, es decir, si da más o
    igual a 0.5 se acepta (regresa un 1) y si no, se rechaza (regresa 0). Si recibe m chips, regresa un
    vector p de m predicciones
    '''
    X_mapeado = mapeoCaracterísticas(X)
    h = sigmoidal(np.dot(X_mapeado, Theta))
    predicciones = []
    for i in range(len(h)):
        if h[i] >= 0.5:
            predicciones.append(1)
        else:
            predicciones.append(0)
    return np.array(predicciones)

'''
Pase a producción, usando el txt proporcionado y
en un main para evitar interferir con la llamada a funciones desde otros archivos.
'''

if __name__ == "__main__":

    # inicializar variables 
    Theta = np.zeros(28)  # Inicializar el vector de parámetros theta con ceros (28 características después del mapeo)
    iteraciones = 70000  # Número de iteraciones para el gradiente descendente
    Lambda = 1  # Valor de lambda para la regularización
    alpha = 0.004  # Tasa de aprendizaje para el gradiente descendente

    # Cargar los datos desde el txt 
    datos = np.loadtxt("Project3/ex2data2.txt", delimiter = ",", unpack=True)

    # extraer datos para primer grafico 
    x1 = datos[0,:]
    x2 = datos[1,:]
    y = datos[2,:]
    X = np.vstack((x1, x2)).T  # Combinar x1 y x2 en una matriz de características X, transpuesta para su lectura

    # Grafico inicial 
    graficaDatos(X, y, Theta)

    # calculamos la función de costo y el gradiente
    J, gradiente = funcionCostoReg(Theta, X, y, Lambda)
    print("Costo inicial:", J) # -> verificado, con Theta en ceros y Lambda 1: 0.693 vs 0.6931471805599453
    print("Gradiente inicial:", gradiente)

    # entrenar el modelo utilizando el gradiente descendente
    Theta = aprende(Theta, X, y, iteraciones)
    print("Theta final:", Theta)

    # hacer predicciones con el modelo entrenado
    predicciones = predice(Theta, X)
    print("Predicciones:", predicciones)

    # graficar los datos con la frontera de decisión cuando Lambda = 1
    graficaDatos(X, y, Theta)

    #  Si todo es correcto, al hacer la predicción sobre el vector X (el vector X aumentado a 28 características), 
    # usando el vector theta que regresó el proceso de optimización, 
    # debe tener una predicción correcta para el 83.050847% de los valores.
    exactitud = np.mean(predicciones == y) * 100
    print("Exactitud del modelo con Lambda = 1:", exactitud, "%") # -> verificado, 83.050847% vs 83.05084745762711 %

    '''
    A continuación, probamos con lambda = 0 y lambda = 100
    '''

    # Probar con lambda = 0
    Lambda = 0
    Theta = np.zeros(28)
    Theta = aprende(Theta, X, y, iteraciones)
    predicciones = predice(Theta, X)
    exactitud = np.mean(predicciones == y) * 100
    graficaDatos(X, y, Theta)
    print("Exactitud del modelo con Lambda = 0:", exactitud, "%") # -> 83.89830508474576 %

    # Probar con lambda = 100
    Lambda = 100
    Theta = np.zeros(28)
    Theta = aprende(Theta, X, y, iteraciones)
    predicciones = predice(Theta, X)
    exactitud = np.mean(predicciones == y) * 100
    graficaDatos(X, y, Theta)
    print("Exactitud del modelo con Lambda = 100:", exactitud, "%") # -> 61.016949152542374 %
    
