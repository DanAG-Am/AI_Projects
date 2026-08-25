'''
Nombre: Amilka Daniela Lopez Aguilar
Matrícula: A01029277
Proyecto 2: Regresión Logística Vectorizada
'''
# importar librerías matemáticas y de graficación
import numpy as np
import matplotlib.pyplot as plt

def graficarDatos(X, y, Theta):
    '''
    Recibe el vector de entradas X, salidas y, y un vector theta. Grafica los
    datos en 2D. Los ejes son las dos calificaciones y pone una X si resultó 
    admitido y una O si no fue admitido. 
    También grafica la recta cuyos parámetros son los del vector theta dado.
    '''
    # grafico de datos
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i, 1], X[i, 2], color='blue', marker='x', label='Admitido' if i == 0 else "")
        else:
            plt.scatter(X[i, 1], X[i, 2], color='red', marker='o', label='No Admitido' if i == 0 else "")
    # graficar la recta de decisión
    x1 = np.array([np.min(X[:, 1]), np.max(X[:, 1])]) # un minimo y un maximo que tomamos como límites y "trazamos el camino" con x2
    x2 = -(Theta[0] + Theta[1] * x1) / Theta[2] # 0 de la sigmoidal, es decir, h_theta = 0.5
    plt.plot(x1, x2, color='green', label='Recta de Decisión')
    plt.xlabel('Examen 1')
    plt.ylabel('Examen 2')
    plt.title('Resultados de Admisión')
    plt.show()

def funcion_sigmoidal(z):
    '''
    Función sigmoidal para la regresión logística.
    Recibe un valor z y devuelve el resultado de la función sigmoidal.
    '''
    return 1 / (1 + np.exp(-z))

def funcionCosto(X, y, Theta):
    '''
    Recibe el vector de entrada X, el de salida y, y un vector theta. Debe
    regresar la función de costo J y el gradiente grad. 
    Es decir, debe regresar las variables [J,grad].
    '''
    m = len(y)
    h = funcion_sigmoidal(np.dot(X, Theta))
    gradiente = (1/m) * np.dot(X.T, (h - y))
    J = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return J, gradiente

def aprende(X, y, Theta, num_iter):
    '''
    Recibe el vector de entrada X, el de salida y, un vector theta
    inicial (el cual puede ser el vector 0) y el número de iteraciones que se correrá el gradiente
    descendente. Debe regresar el vector de parámetros theta, encontrado por gradiente
    descendente. Si todo es correcto, al mandar a llamar a funcionCosto con el vector theta
    encontrado, el resultado debe ser alrededor de 0.203.
    '''
    alpha = 0.0041
    for i in range(num_iter):
        J, gradiente = funcionCosto(X, y, Theta)
        Theta = Theta - alpha * gradiente
    return Theta

def predice(X, Theta):
    '''
    Recibe un vector theta y un vector X para varios estudiantes. Regresa el vector
    p de predicción sobre su aceptación utilizando un valor de umbral natural de 0.5, es decir, si da
    más o igual a 0.5 se acepta (regresa un 1) y si no, se rechaza (regresa 0). Si recibe m estudiantes,
    regresa un vector p de m predicciones. Si todo es correcto, un estudiante con examen 1 de 45 y
    examen 2 de 85, tendrá una probabilidad de ser admitido de 0.774 y entonces predice 1
    '''
    h = funcion_sigmoidal(np.dot(X, Theta))
    predicciones = []
    for i in range(len(h)):
        if h[i] >= 0.5:
            predicciones.append(1)
        else:
            predicciones.append(0)
    return np.array(predicciones)

'''
Pase a producción, usando el csv proporcionado y
en un main para evitar interferir con la llamada a funciones desde otros archivos.
'''

if __name__ == "__main__":

    # cargamos el archivo de prueba
    datos = np.loadtxt('ML Algorithms/Project2/ex2data1.txt', delimiter=',', unpack=True)

    # extraer columnas de X y, poner la columna de 1's en X para vectorización
    X = np.array([np.ones(len(datos[0])), datos[0], datos[1]]).T # -> x = 1 (de longitud x1 y por ende de x2) x1 x2 transpuesta para lectura y extracción
    y = datos[2]

    # inicializamos theta en 0
    Theta = np.array([0, 0, 0])

    #graficamos los datos
    graficarDatos(X, y, Theta)

    # calculamos la función de costo y el gradiente
    J, gradiente = funcionCosto(X, y, Theta)

    print("Costo inicial: ", J)
    print("Gradiente inicial: ", gradiente)

    # aprendemos los parámetros theta
    Theta = aprende(X, y, Theta, 780000)
    print("Theta final: ", Theta)

    # calculamos la función de costo y el gradiente con los nuevos parámetros
    J, gradiente = funcionCosto(X, y, Theta)
    print("Costo final: ", J) # verificado -> 0.203 vs 0.20350876005795487
    print("Gradiente final: ", gradiente)

    # graficamos los datos con la recta de decisión
    graficarDatos(X, y, Theta)

    # probamos la predicción con un estudiante de examen 1 = 45 y examen 2 = 85
    estudiante = np.array([[1, 45, 85]])
    prediccion = predice(estudiante, Theta)
    print("Predicción para estudiante con examen 1 = 45 y examen 2 = 85: ", prediccion[0]) # verificado -> 1 vs 1
    print("Probabilidad de ser admitido: ", funcion_sigmoidal(np.dot(estudiante, Theta))) # verificado -> 0.774 vs 0.77403296

'''
¿Es necesario agregar la columna de 1s a X para facilitar la vectorización?

Sí, ya que esto permite tener un termine constante y 
formar operaciones con matrices de esta forma, por ejemplo:
z = Xθ, lo que en código se traduce a np.dot(X, Theta), 
en lugar de usar fors y hacer un código lento y redundante.
'''