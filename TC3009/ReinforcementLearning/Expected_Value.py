'''
A01029277 - Amilka Daniela Lopez Aguilar
Agente escoge el mejor slot on RL 
    Slot 1 - valor esperado 7.5: - 75% paga 10, 25% paga 0
    Slot 2 - valor esperado 2.5: - 75% paga 0, 25% paga 10
Entrenamos al agente usando aprendizaje por refuerzo, vamos haciendo la actualizacion de los parámetros.
Correr 10 veces. 
Graficar el aprendizaje del agente.
'''

import numpy as np
import matplotlib.pyplot as plt

lr = 0.1
num_iteraciones = 50

for i in range(10):
    v = np.zeros(2)
    h = []
    for _ in range(num_iteraciones):
        p = np.exp(v) / np.exp(v).sum()
        a = np.random.choice(2, p=p)
        pago = np.random.choice([10, 0] if a == 0 else [0, 10], p=[0.75, 0.25])
        v[a] += lr * (pago - v[a])
        h.append(v.copy())
    h = np.array(h)
    plt.plot(h[:, 0], color='C0', alpha=0.5, label='Slot 1' if i == 0 else None)
    plt.plot(h[:, 1], color='C1', alpha=0.5, label='Slot 2' if i == 0 else None)
plt.xlabel('Iteración')
plt.ylabel('Valor estimado')
plt.legend()
plt.show()

# probabilidad de escoger cada slot (última corrida)
p = np.exp(h) / np.exp(h).sum(axis=1, keepdims=True)
plt.plot(p[:, 0], label='Probabilidad Slot 1')
plt.plot(p[:, 1], label='Probabilidad Slot 2')
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

v = np.zeros(2)
baseline = 0
h = []
hb = []
for k in range(1, num_iteraciones + 1):
    p = np.exp(v) / np.exp(v).sum()
    a = np.random.choice(2, p=p)
    pago = np.random.choice([10, 0] if a == 0 else [0, 10], p=[0.75, 0.25])
    v[a] += lr * (pago - v[a])
    baseline += (pago - baseline) / k
    h.append(v.copy())
    hb.append(baseline)
h = np.array(h)
plt.plot(h[:, 0], label='Slot 1')
plt.plot(h[:, 1], label='Slot 2')
plt.plot(hb, label='Baseline')
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
Valor esperado con aproximación gradual: Q += lr * (reward - Q)
10000 iteraciones, lr = 1/k, 0.1, 0.01, 0.001
'''
casos = [([0, 10], [0.6, 0.4]), ([0, 100], [0.97, 0.03])]
for opciones, probs in casos:
    for lr_nombre in ['1/k', 0.1, 0.01, 0.001]:
        q = 0
        historial = []
        for k in range(1, 10001):
            pago = np.random.choice(opciones, p=probs)
            q += (1 / k if lr_nombre == '1/k' else lr_nombre) * (pago - q)
            historial.append(q)
        plt.plot(historial, label=f'lr = {lr_nombre}')
    plt.axhline(np.dot(opciones, probs), color='k', linestyle='--', label='valor esperado')
    plt.title(f'opciones {opciones}, probabilidades {probs}')
    plt.xlabel('Iteración')
    plt.ylabel('Valor esperado estimado')
    plt.legend()
    plt.show()
