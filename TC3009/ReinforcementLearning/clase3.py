'''
Agente escoge el mejor slot on RL 
    Slot 1 - valor esperado 7.5: - 75% paga 10, 25% paga 0
    Slot 2 - valor esperado 2.5: - 75% paga 0, 25% paga 10
Entrenamos al agente usando aprendizaje por refuerzo, vamos haciendo la actualizacion de los parámetros.
Correr 10 veces. 
Graficar el aprendizaje del agente.
'''

from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt

slot_1 = [10, 0]  # Slot 1: 75% paga 10, 25% paga 0
slot_2 = [0, 10]  # Slot 2: 75% paga 0, 25% paga 10

# Probabilidades de pago de cada slot
prob_slot_1 = [0.75,0.25]
prob_slot_2 = [0.75,0.25]

# parametros 
lr = 0.1  # Tasa de aprendizaje
num_iteraciones = 50  # Número de iteraciones de entrenamiento
valor_slot_1 = 0  # Valor inicial del slot 1
valor_slot_2 = 0  # Valor inicial del slot 2

def elegir_slot(valor_slot_1, valor_slot_2):
    valores = np.array([valor_slot_1, valor_slot_2])
    probabilidades = softmax(valores)
    return np.random.choice([1, 2], p=probabilidades)

def muestreo(probabilidades, pagos):
    return np.random.choice(pagos, p=probabilidades)

def actualizar_valor(valor_actual, pago, lr):
    return valor_actual + lr * (pago - valor_actual)

# Entrenamiento del agente
historial_valor_slot_1 = []
historial_valor_slot_2 = []
for _ in range(num_iteraciones):
    slot_elegido = elegir_slot(valor_slot_1, valor_slot_2)
    if slot_elegido == 1:
        pago = muestreo(prob_slot_1, slot_1)
        valor_slot_1 = actualizar_valor(valor_slot_1, pago, lr)
    else:
        pago = muestreo(prob_slot_2, slot_2)
        valor_slot_2 = actualizar_valor(valor_slot_2, pago, lr)
    historial_valor_slot_1.append(valor_slot_1)
    historial_valor_slot_2.append(valor_slot_2)

# Graficar el aprendizaje del agente
plt.plot(historial_valor_slot_1, label='Slot 1')
plt.plot(historial_valor_slot_2, label='Slot 2')
plt.xlabel('Iteración')
plt.ylabel('Valor estimado')
plt.legend()
plt.show()

# volvemos a correr
valor_slot_1 = 0  # Reiniciar valor inicial del slot 1
valor_slot_2 = 0  # Reiniciar valor inicial del slot 2
historial_valor_slot_1 = []
historial_valor_slot_2 = []
for _ in range(num_iteraciones):
    slot_elegido = elegir_slot(valor_slot_1, valor_slot_2)
    if slot_elegido == 1:
        pago = muestreo(prob_slot_1, slot_1)
        valor_slot_1 = actualizar_valor(valor_slot_1, pago, lr)
    else:
        pago = muestreo(prob_slot_2, slot_2)
        valor_slot_2 = actualizar_valor(valor_slot_2, pago, lr)
    historial_valor_slot_1.append(valor_slot_1)
    historial_valor_slot_2.append(valor_slot_2)

# Graficar el aprendizaje del agente nuevamente
plt.plot(historial_valor_slot_1, label='Slot 1')
plt.plot(historial_valor_slot_2, label='Slot 2')
plt.xlabel('Iteración')
plt.ylabel('Valor estimado')
plt.legend()
plt.show()

# volvemos a correr
valor_slot_1 = 0  # Reiniciar valor inicial del slot 1
valor_slot_2 = 0  # Reiniciar valor inicial del slot 2
historial_valor_slot_1 = []
historial_valor_slot_2 = []
for _ in range(num_iteraciones):
    slot_elegido = elegir_slot(valor_slot_1, valor_slot_2)
    if slot_elegido == 1:
        pago = muestreo(prob_slot_1, slot_1)
        valor_slot_1 = actualizar_valor(valor_slot_1, pago, lr)
    else:
        pago = muestreo(prob_slot_2, slot_2)
        valor_slot_2 = actualizar_valor(valor_slot_2, pago, lr)
    historial_valor_slot_1.append(valor_slot_1)
    historial_valor_slot_2.append(valor_slot_2)

# Graficar el aprendizaje del agente nuevamente
plt.plot(historial_valor_slot_1, label='Slot 1')
plt.plot(historial_valor_slot_2, label='Slot 2')
plt.xlabel('Iteración')
plt.ylabel('Valor estimado')
plt.legend()
plt.show()

# volvemos a correr
valor_slot_1 = 0  # Reiniciar valor inicial del slot 1
valor_slot_2 = 0  # Reiniciar valor inicial del slot 2
historial_valor_slot_1 = []
historial_valor_slot_2 = []
for _ in range(num_iteraciones):
    slot_elegido = elegir_slot(valor_slot_1, valor_slot_2)
    if slot_elegido == 1:
        pago = muestreo(prob_slot_1, slot_1)
        valor_slot_1 = actualizar_valor(valor_slot_1, pago, lr)
    else:
        pago = muestreo(prob_slot_2, slot_2)
        valor_slot_2 = actualizar_valor(valor_slot_2, pago, lr)
    historial_valor_slot_1.append(valor_slot_1)
    historial_valor_slot_2.append(valor_slot_2)

# Graficar el aprendizaje del agente nuevamente
plt.plot(historial_valor_slot_1, label='Slot 1')
plt.plot(historial_valor_slot_2, label='Slot 2')
plt.xlabel('Iteración')
plt.ylabel('Valor estimado')
plt.legend()
plt.show()

# volvemos a correr
valor_slot_1 = 0  # Reiniciar valor inicial del slot 1
valor_slot_2 = 0  # Reiniciar valor inicial del slot 2
historial_valor_slot_1 = []
historial_valor_slot_2 = []
for _ in range(num_iteraciones):
    slot_elegido = elegir_slot(valor_slot_1, valor_slot_2)
    if slot_elegido == 1:
        pago = muestreo(prob_slot_1, slot_1)
        valor_slot_1 = actualizar_valor(valor_slot_1, pago, lr)
    else:
        pago = muestreo(prob_slot_2, slot_2)
        valor_slot_2 = actualizar_valor(valor_slot_2, pago, lr)
    historial_valor_slot_1.append(valor_slot_1)
    historial_valor_slot_2.append(valor_slot_2)

# Graficar el aprendizaje del agente nuevamente
plt.plot(historial_valor_slot_1, label='Slot 1')
plt.plot(historial_valor_slot_2, label='Slot 2')
plt.xlabel('Iteración')
plt.ylabel('Valor estimado')
plt.legend()
plt.show()

# graficar probabilidad de escoger cada slot
plt.plot([softmax(np.array([v1, v2]))[0] for v1, v2 in zip(historial_valor_slot_1, historial_valor_slot_2)], label='Probabilidad Slot 1')
plt.plot([softmax(np.array([v1, v2]))[1] for v1, v2 in zip(historial_valor_slot_1, historial_valor_slot_2)], label='Probabilidad Slot 2')
plt.xlabel('Iteración')
plt.ylabel('Probabilidad de escoger')
plt.legend()
plt.show()

'''
como son variables aleatorias, los resultados pueden variar en cada ejecución
lo que implica que a veces el agente puede elegir un slot subóptimo.
ya que la varianza de los rewards puede afectar la estimación del valor de cada slot.
Esto lo podemos corregir con baseline.
'''

'''
Ejercicio con baseline, usando promedio de reward
Corriendo el algoritmo 10 veces seguidas
'''

valor_slot_1 = 0  # Reiniciar valor inicial del slot 1
valor_slot_2 = 0  # Reiniciar valor inicial del slot 2
historial_valor_slot_1 = []
historial_valor_slot_2 = []
historial_baseline = []  # Inicializar historial del baseline antes del bucle
baseline = 0  # Inicializar el baseline
for _ in range(num_iteraciones):
    slot_elegido = elegir_slot(valor_slot_1, valor_slot_2)
    if slot_elegido == 1:
        pago = muestreo(prob_slot_1, slot_1)
        valor_slot_1 = actualizar_valor(valor_slot_1, pago, lr)
    else:
        pago = muestreo(prob_slot_2, slot_2)
        valor_slot_2 = actualizar_valor(valor_slot_2, pago, lr)
    baseline = (baseline * len(historial_valor_slot_1) + pago) / (len(historial_valor_slot_1) + 1)
    historial_valor_slot_1.append(valor_slot_1)
    historial_valor_slot_2.append(valor_slot_2)
    historial_baseline.append(baseline)

# Inicializar historial_baseline antes del bucle
historial_baseline = []

# Graficar la evolución del baseline
plt.plot(historial_baseline, label='Baseline')
plt.xlabel('Iteración')
plt.ylabel('Valor del baseline')
plt.legend()
plt.show()

# Graficar el aprendizaje del agente con baseline
plt.plot(historial_valor_slot_1, label='Slot 1')
plt.plot(historial_valor_slot_2, label='Slot 2')
plt.plot(historial_baseline, label='Baseline')
plt.xlabel('Iteración')
plt.ylabel('Valor estimado')
plt.legend()
plt.show()

'''
3 formas de reducir la varianza en el aprendizaje por refuerzo por medio del baseline
1. Usar el promedio de los rewards como baseline.
2. Reward - promedio de los rewards anteriores
3. Reward - Reward ganado con esa acción específica
'''

'''
Códifo con
Primer baseline 
V(s) = promedio de los rewards obtenidos en el estado s
'''
# Implementación del primer baseline usando el promedio de los rewards obtenidos en el estado s
valor_slot_1 = 0  # Reiniciar valor inicial del slot 1
valor_slot_2 = 0  # Reiniciar valor inicial del slot 2
historial_valor_slot_1 = []
historial_valor_slot_2 = []
historial_baseline = []  # Inicializar historial del baseline antes del bucle
baseline = 0  # Inicializar el baseline
for _ in range(num_iteraciones):
    slot_elegido = elegir_slot(valor_slot_1, valor_slot_2)
    if slot_elegido == 1:
        pago = muestreo(prob_slot_1, slot_1)
        valor_slot_1 = actualizar_valor(valor_slot_1, pago, lr)
    else:
        pago = muestreo(prob_slot_2, slot_2)
        valor_slot_2 = actualizar_valor(valor_slot_2, pago, lr)
    baseline = (baseline * len(historial_valor_slot_1) + pago) / (len(historial_valor_slot_1) + 1)
    historial_valor_slot_1.append(valor_slot_1)
    historial_valor_slot_2.append(valor_slot_2)
    historial_baseline.append(baseline)

# Graficar la evolución del baseline
plt.plot(historial_baseline, label='Baseline')
plt.xlabel('Iteración')
plt.ylabel('Valor del baseline')
plt.legend()
plt.show()

'''
Codifo con segundo baseline
Q(a,s) = promedio de los rewards obtenidos al tomar la acción a en el estado s  
'''
valor_slot_1 = 0
valor_slot_2 = 0
historial_valor_slot_1 = []
historial_valor_slot_2 = []
historial_baseline = []  # Inicializar historial del baseline antes del bucle
baseline = 0  # Inicializar el baseline
for _ in range(num_iteraciones):
    slot_elegido = elegir_slot(valor_slot_1, valor_slot_2)
    if slot_elegido == 1:
        pago = muestreo(prob_slot_1, slot_1)
        valor_slot_1 = actualizar_valor(valor_slot_1, pago, lr)
    else:
        pago = muestreo(prob_slot_2, slot_2)
        valor_slot_2 = actualizar_valor(valor_slot_2, pago, lr)
    baseline = (baseline * len(historial_valor_slot_1) + pago) / (len(historial_valor_slot_1) + 1)
    historial_valor_slot_1.append(valor_slot_1)
    historial_valor_slot_2.append(valor_slot_2)
    historial_baseline.append(baseline)

# Graficar la evolución del baseline
plt.plot(historial_baseline, label='Baseline')
plt.xlabel('Iteración')
plt.ylabel('Valor del baseline')
plt.legend()
plt.show()

'''
Codifo con tercer baseline
Q(aus)+= lambda * (reward - Q(aus))
'''
valor_slot_1 = 0
valor_slot_2 = 0
historial_valor_slot_1 = []
historial_valor_slot_2 = []
historial_baseline = []  # Inicializar historial del baseline antes del bucle
baseline = 0  # Inicializar el baseline
for _ in range(num_iteraciones):
    slot_elegido = elegir_slot(valor_slot_1, valor_slot_2)
    if slot_elegido == 1:
        pago = muestreo(prob_slot_1, slot_1)
        valor_slot_1 = actualizar_valor(valor_slot_1, pago, lr)
    else:
        pago = muestreo(prob_slot_2, slot_2)
        valor_slot_2 = actualizar_valor(valor_slot_2, pago, lr)
    baseline = baseline + lr * (pago - baseline)
    historial_valor_slot_1.append(valor_slot_1)
    historial_valor_slot_2.append(valor_slot_2)
    historial_baseline.append(baseline)

# Graficar la evolución del baseline
plt.plot(historial_baseline, label='Baseline')
plt.xlabel('Iteración')
plt.ylabel('Valor del baseline')
plt.legend()
plt.show()
