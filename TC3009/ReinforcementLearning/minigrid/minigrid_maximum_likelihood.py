"""
A01029277 - Amilka Daniela Lopez Aguilar
Estimacion probabilistica de la distribucion de acciones en trayectorias de agentes
La politica usa softmax(log(conteos + 1)): frecuencias suavizadas,
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from minigrid.minigrid_simple_env import SimpleEnv, verify_rewards

EPISODIOS = 3000
REINICIAR_CONTEOS = True
SOLO_DEMOSTRAR = False
SOLO_VERIFICAR = False
MOSTRAR_AMBIENTE = True
MOSTRAR_GRAFICAS = True
ARCHIVO_CONTEOS = Path(__file__).with_name("minigrid_action_counts.txt")

class MaximumLikelihoodAgent:
    def __init__(self, env, seed=None):
        self.env = env
        self.actions = (env.actions.left, env.actions.right, env.actions.forward)
        self.counts = np.zeros((env.width * env.height * 4, len(self.actions)))
        self.rng = np.random.default_rng(seed) # elección random de acciones (0,1,2 definirlo más adelante)
        self.rewards = []
        self.successes = []

    def state(self):
        pos = self.env.agent_pos
        # Cada posicion y orientacion corresponde a una fila de la matriz.
        return (int(pos[0]) * self.env.height + int(pos[1])) * 4 + self.env.agent_dir

    def probabilities(self, state):
        # Equivale a (conteos + 1) / suma(conteos + 1); permite explorar acciones no vistas.
        return softmax(np.log(self.counts[state] + 1.0))

    def choose_action(self, state, demonstrate=False):
        probabilities = self.probabilities(state)
        # Al demostrar elige un maximo; al entrenar muestrea segun las probabilidades.
        if demonstrate:
            best = []
            for action_index, probability in enumerate(probabilities):
                if probability == max(probabilities):
                    best.append(action_index)
            return int(self.rng.choice(best))
        return int(self.rng.choice(len(self.actions), p=probabilities))

    def update(self, trajectory):
        path = []
        states = []
        for state, action_index in trajectory:
            if state in states:
                # Volver al mismo estado forma un ciclo: quitamos ese tramo del recorrido.
                start = states.index(state)
                path = path[:start]
                states = states[:start]
            states.append(state)
            path.append((state, action_index))
        for state, action_index in path:
            self.counts[state, action_index] += 1

    def print_state(self, state):
        print(f"Estado {state}: probabilidades={np.round(self.probabilities(state), 3)}")
        print("Orden: izquierda, derecha, avanzar.")

    def save(self, path):
        path = Path(path)
        np.savetxt(path, self.counts, fmt="%d")
        probabilities = []
        for state in range(len(self.counts)):
            probabilities.append(self.probabilities(state))
        probability_path = path.with_name(path.stem + "_probabilities.txt")
        np.savetxt(probability_path, probabilities)
        print("Conteos y probabilidades guardados.")

    def load(self, path):
        self.counts = np.loadtxt(path)
        print("Conteos cargados.")

    def train(self, episodes=3000):
        if episodes <= 0:
            print("El numero de episodios debe ser positivo.")
            return
        self.rewards.clear()
        self.successes.clear()
        for episode in range(episodes):
            self.env.reset()
            trajectory = []
            total_reward = 0.0
            while True:
                state = self.state()
                action_index = self.choose_action(state)
                trajectory.append((state, action_index))
                _, reward, terminated, truncated, _ = self.env.step(self.actions[action_index])
                total_reward += reward
                if terminated or truncated:
                    break
            success = terminated and reward == 10.0
            if success:
                # Solo los recorridos que alcanzan la meta aumentan los conteos.
                self.update(trajectory)
            self.rewards.append(total_reward)
            self.successes.append(success)
            if (episode + 1) % 100 == 0:
                print(f"Episodio {episode + 1}/{episodes}: "
                      f"recompensa promedio={np.mean(self.rewards[-100:]):.2f}")
        print(f"Recompensa promedio: {np.mean(self.rewards):.2f}")
        if not any(self.successes):
            print("No se alcanzo la meta. Los conteos no cambiaron.")

    def run(self):
        self.env.reset()
        self.print_state(self.state())
        while True:
            action_index = self.choose_action(self.state(), demonstrate=True)
            _, reward, terminated, truncated, _ = self.env.step(self.actions[action_index])
            if terminated or truncated:
                break
        reached_goal = terminated and reward == 10.0
        print(f"Meta alcanzada: {'si' if reached_goal else 'no'}. Pasos: {self.env.step_count}")


def main():
    if not verify_rewards():
        print("No se pudo verificar el ambiente.")
        return
    if SOLO_VERIFICAR:
        return
    if EPISODIOS <= 0:
        print("El numero de episodios debe ser positivo.")
        return
    if SOLO_DEMOSTRAR and not ARCHIVO_CONTEOS.is_file():
        print("No existe una matriz guardada para demostrar.")
        return
    env = SimpleEnv(max_steps=1000)
    agent = MaximumLikelihoodAgent(env)
    if ARCHIVO_CONTEOS.is_file() and (SOLO_DEMOSTRAR or not REINICIAR_CONTEOS):
        agent.load(ARCHIVO_CONTEOS)
    else:
        print("Matriz nueva en ceros.")
    if not SOLO_DEMOSTRAR:
        agent.train(EPISODIOS)
        agent.save(ARCHIVO_CONTEOS)
    env.close()
    env = SimpleEnv(render_mode="human" if MOSTRAR_AMBIENTE else None, max_steps=200)
    agent.env = env
    agent.run()
    env.close()
    if SOLO_DEMOSTRAR or not MOSTRAR_GRAFICAS:
        return
    window = min(100, len(agent.rewards))
    episodes = []
    average_rewards = []
    for end in range(window, len(agent.rewards) + 1):
        start = end - window
        episodes.append(end)
        average_rewards.append(np.mean(agent.rewards[start:end]))

    plt.figure(figsize=(10, 7))
    plt.plot(episodes, average_rewards, label="Reward AVG")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()