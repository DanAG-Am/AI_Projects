
'''
A01029277 - Amilka Daniela Lopez Aguilar
'''
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.minigrid_env import MiniGridEnv

EPISODIOS = 6000
SOLO_DEMOSTRAR = False
SOLO_VERIFICAR = False
MOSTRAR_AMBIENTE = True
MOSTRAR_GRAFICAS = True
ARCHIVO_POLITICA = Path(__file__).with_name("minigrid_policy_parameters.txt")

historial_rewards = []

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=19,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size
        self.key_positions = []
        self.lava_positions = []
        self.start_agent_pos = (random.randint(1, size - 2), random.randint(1, size - 2))

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for y in range(1, height - 1):
            self.put_obj(Wall(), width // 2, y)

        for x in range(1, width - 1):
            self.put_obj(Wall(), x, height // 2)

        openings = [
            (width // 2, 5),
            (width // 2, 15),
            (5, height // 2),
            (15, height // 2),
        ]

        for x, y in openings:
            self.grid.set(x, y, None)

        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        self._place_agent()

        self.mission = "Reach the goal"

    def _place_agent(self):
        while True:
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)
            pos = (x, y)

            if self.grid.get(*pos) is None and pos != self.goal_pos:
                self.agent_pos = pos
                self.agent_dir = random.randint(0, 3)
                break

    def reset(self, **kwargs):
        self.stepped_floors = set()
        obs = super().reset(**kwargs)
        return obs

    def step(self, action):
        prev_pos = self.agent_pos
        prev_dir = self.agent_dir
        obs, reward, terminated, truncated, info = super().step(action)

        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = 10.0
            terminated = True
        elif prev_dir == self.agent_dir and prev_pos == self.agent_pos:
            reward = -0.3
        else:
            reward = -0.01

        return obs, reward, terminated, truncated, info

    class Agent():
        def __init__(self, env, parameters=None):
            self.env = env
            self.actions = (env.actions.left, env.actions.right, env.actions.forward)
            shape = (env.width * env.height * 4, len(self.actions))
            self.parameters = np.random.uniform(-0.1, 0.1, size=shape) if parameters is None else np.array(parameters, dtype=float, copy=True)
            if not self.validate_parameters():
                raise ValueError("La matriz de parametros tiene dimensiones o valores no validos.")
            self.reached_goals = []
            self.episode_steps = []
            self.update_position()

        def validate_parameters(self):
            expected = (self.env.width * self.env.height * 4, len(self.actions))
            return self.parameters.shape == expected and np.isfinite(self.parameters).all()

        def save(self, path):
            if not self.validate_parameters():
                print("Matriz no valida. No se guardo.")
                return
            np.savetxt(
                path,
                self.parameters,
                fmt="%.18e",
                header=f"REINFORCE: width={self.env.width}, height={self.env.height}; "
                       "state=(x*height+y)*4+direction; actions=left,right,forward"
            )
            restored = np.loadtxt(path, ndmin=2)
            if not np.array_equal(self.parameters, restored):
                print("Los valores guardados no coinciden.")
                return
            print("Matriz guardada.")

        def load(self, path):
            previous = self.parameters
            self.parameters = np.loadtxt(path, ndmin=2)
            if not self.validate_parameters():
                self.parameters = previous
                print("Matriz no valida. Se conserva la anterior.")
                return
            print("Matriz cargada.")

        def update_position(self):
            self.position = self.env.agent_pos
            self.direction = self.env.agent_dir

        def state(self):
            return (int(self.position[0]) * self.env.height + int(self.position[1])) * 4 + self.direction

        def action_probabilities(self, state=None):
            if state is None:
                state = self.state()
            values = self.parameters[state]
            logits = values - np.max(values)
            weights = np.exp(logits)
            return weights / weights.sum()

        def choose_action(self, deterministic=False):
            probabilities = self.action_probabilities()
            if deterministic:
                return int(np.argmax(probabilities))
            return int(np.random.choice(len(self.actions), p=probabilities))

        def act(self, action=None):
            if action is None:
                action = self.actions[self.choose_action()]
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.update_position()
            return obs, reward, terminated, truncated, info

        def update_policy(self, trajectory, learning_rate, discount):
            discounted_return = 0.0
            returns = np.zeros(len(trajectory), dtype=float)

            for step in reversed(range(len(trajectory))):
                discounted_return = trajectory[step][2] + discount * discounted_return
                returns[step] = discounted_return

            for step, (state, action_index, _, probabilities) in enumerate(trajectory):
                gradient = -probabilities.copy()
                gradient[action_index] += 1.0
                self.parameters[state] += (
                    learning_rate * discount**step * returns[step] * gradient
                )

        def train(self, episodes=6000, learning_rate=0.01, discount=0.99):
            if episodes <= 0:
                print("El numero de episodios debe ser positivo.")
                return

            historial_rewards.clear()
            self.reached_goals.clear()
            self.episode_steps.clear()

            for episode in range(episodes):
                self.env.reset()
                self.update_position()
                episode_reward = 0
                trajectory = []

                while True:
                    state = self.state()
                    probabilities = self.action_probabilities(state)
                    action_index = int(np.random.choice(len(self.actions), p=probabilities))
                    _, reward, terminated, truncated, _ = self.act(self.actions[action_index])
                    episode_reward += reward
                    trajectory.append((state, action_index, reward, probabilities))

                    if terminated or truncated:
                        break

                self.update_policy(trajectory, learning_rate, discount)
                historial_rewards.append(episode_reward)
                self.reached_goals.append(terminated and reward == 10.0)
                self.episode_steps.append(self.env.step_count)

                if (episode + 1) % 100 == 0:
                    print(
                        f"Episodio {episode + 1}/{episodes}: "
                        f"recompensa promedio={np.mean(historial_rewards[-100:]):.2f}"
                    )

            print(f"Recompensa promedio: {np.mean(historial_rewards):.2f}")

        def run(self):
            self.env.reset()
            self.update_position()

            while True:
                action_index = self.choose_action()
                _, reward, terminated, truncated, _ = self.act(self.actions[action_index])

                if terminated or truncated:
                    return reward, terminated and reward == 10.0


def verify_rewards():
    env = SimpleEnv(max_steps=2)
    cases = [
        ("Llegar a la meta", (16, 17), 0, env.actions.forward, 10.0),
        ("Chocar contra pared", (1, 1), 2, env.actions.forward, -0.3),
        ("Avanzar", (1, 1), 0, env.actions.forward, -0.01),
        ("Girar izquierda", (1, 1), 0, env.actions.left, -0.01),
        ("Girar derecha", (1, 1), 0, env.actions.right, -0.01),
    ]

    print("Verificacion de recompensas:")

    for label, position, direction, action, expected in cases:
        env.reset()
        env.agent_pos = position
        env.agent_dir = direction
        _, reward, terminated, truncated, _ = env.step(action)

        valid = (
            isinstance(reward, (float, np.floating))
            and np.isclose(reward, expected)
            and terminated == (expected == 10.0)
            and not truncated
        )

        print(f"{label}: {reward:+.2f} ({'correcto' if valid else 'incorrecto'})")

        if not valid:
            env.close()
            return False

    env.reset()
    env.agent_pos = (1, 1)
    env.agent_dir = 0
    env.step(env.actions.forward)
    _, reward, terminated, truncated, _ = env.step(env.actions.forward)

    valid = np.isclose(reward, -0.01) and truncated and not terminated
    print(f"Limite de pasos: {reward:+.2f} ({'correcto' if valid else 'incorrecto'})")

    env.close()

    if not valid:
        return False

    print("Recompensas verificadas.")
    return True


def main():
    if not verify_rewards():
        print("No se pudo verificar el ambiente.")
        return

    if SOLO_VERIFICAR:
        return

    if not SOLO_DEMOSTRAR and EPISODIOS <= 0:
        print("El numero de episodios debe ser positivo.")
        return

    if SOLO_DEMOSTRAR and not ARCHIVO_POLITICA.is_file():
        print("No existe una matriz guardada para demostrar.")
        return

    training_env = SimpleEnv(max_steps=1000)
    training_env.reset()
    agent = SimpleEnv.Agent(training_env)

    if ARCHIVO_POLITICA.is_file():
        agent.load(ARCHIVO_POLITICA)
    else:
        print("Matriz nueva en ceros.")

    if not SOLO_DEMOSTRAR:
        agent.train(episodes=EPISODIOS)
        agent.save(ARCHIVO_POLITICA)

    reached_goals = agent.reached_goals.copy()
    training_env.close()

    env = SimpleEnv(
        render_mode="human" if MOSTRAR_AMBIENTE else None,
        max_steps=200
    )
    agent = SimpleEnv.Agent(env, parameters=agent.parameters)
    before_demo = agent.parameters.copy()
    reward, reached_goal = agent.run()

    if not np.array_equal(before_demo, agent.parameters):
        print("La demostracion modifico la matriz de parametros.")
        env.close()
        return

    print(f"Meta alcanzada: {'si' if reached_goal else 'no'}. Pasos: {env.step_count}")
    env.close()

    if SOLO_DEMOSTRAR or not MOSTRAR_GRAFICAS:
        return

    window = min(100, len(historial_rewards))
    episodes = []
    average_rewards = []
    success_rate = []

    for end in range(window, len(historial_rewards) + 1):
        start = end - window
        episodes.append(end)
        average_rewards.append(np.mean(historial_rewards[start:end]))
        success_rate.append(100 * np.mean(reached_goals[start:end]))

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.plot(episodes, average_rewards, label="Reward AVG")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.show()

if __name__ == "__main__":
    main()
