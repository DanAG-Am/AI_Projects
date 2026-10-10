from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=19,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size
        self.start_agent_pos=(random.randint(1, size - 2), random.randint(1, size - 2))
        
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
        
        # Place walls in straight lines
        # Vertical walls
        for y in range(1, height-1):
            self.put_obj(Wall(), width // 2, y)
        
        # Horizontal walls
        for x in range(1, width-1):
            self.put_obj(Wall(), x, height//2)
        
        # Create openings in the walls
        openings = [(width//2,5),(width//2,15),(5,height//2),(15,height//2),]
        
        for x, y in openings:
            self.grid.set(x, y, None)

        
        # Place a goal square in the bottom-right corner
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)
        
        self._place_agent()
        
        self.mission = "Reach the goal"

    def _place_agent(self):
        '''
        Quitamos la colocación aleatoria del agente y lo colocamos 
        en la posición inicial fija.
        '''
        self.agent_pos = self.start_agent_pos
        self.agent_dir = 0

    def state(self):
        """Necesitamos el estado completo del mapa fijo: posicion y orientacion."""
        return (int(self.agent_pos[0]), int(self.agent_pos[1]), int(self.agent_dir))

    def step(self, action):
        prev_pos = self.agent_pos
        prev_dir = self.agent_dir
        obs, reward, terminated, truncated, info = super().step(action)

        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = 10.0
            terminated = True
            print("GOAL REACHED!") # -> cambiamos el orden para priorizar, con print por ahora
        elif prev_dir == self.agent_dir and prev_pos == tuple(self.agent_pos):
            reward = -0.3
        else:
            reward = -0.01
        return obs, reward, terminated, truncated, info


def state_index(env):
    agent_x, agent_y, direction = env.state()
    return (agent_y * env.width + agent_x) * 4 + direction


def train(env, episodes=3000):
    """Acciones utilizadas: 0 = izquierda, 1 = derecha, 2 = avanzar."""
    q_table = np.zeros((env.width * env.height * 4, env.action_space.n))
    alpha = 0.2
    gamma = 0.99
    epsilon = 1.0
    random.seed(7)

    for episode in range(episodes):
        env.reset(seed=7)
        state = state_index(env)
        total_reward = 0.0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                action = int(np.argmax(q_table[state, :3]))

            _, reward, terminated, truncated, _ = env.step(action)
            next_state = state_index(env)

            if terminated:
                future_value = 0.0
            else:
                future_value = np.max(q_table[next_state, :3])

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * future_value - q_table[state, action]
            )

            state = next_state
            total_reward += reward
            done = terminated or truncated

        epsilon = max(0.05, epsilon * 0.997)
        if (episode + 1) % 100 == 0:
            print(f"Episodio {episode + 1}: recompensa = {total_reward:.2f}")

    return q_table


def demonstrate(env, q_table):
    """Elige siempre la mejor accion y no modifica la tabla Q."""
    env.reset(seed=7)
    done = False
    total_reward = 0.0

    while not done:
        state = state_index(env)
        action = int(np.argmax(q_table[state, :3]))
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    reached_goal = terminated and reward == 10.0
    print(f"Meta alcanzada: {reached_goal}")
    print(f"Pasos: {env.step_count}, recompensa: {total_reward:.2f}")
    return reached_goal


def main():
    parser = argparse.ArgumentParser(description="Q-Learning en MiniGrid")
    parser.add_argument("--demo", action="store_true",
                        help="Mostrar lo aprendido sin entrenar")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--headless", action="store_true",
                        help="Ejecutar sin ventana")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("Los episodios deben ser positivos")

    path = Path(__file__).with_name("minigrid_q_table.npy")
    if args.demo:
        if not path.exists():
            parser.error("Primero ejecuta el programa sin --demo para entrenar")
        q_table = np.load(path)
    else:
        env = SimpleEnv()
        q_table = train(env, args.episodes)
        np.save(path, q_table)
        env.close()
        print("Tabla Q guardada")

    if args.headless:
        env = SimpleEnv()
    else:
        env = SimpleEnv(render_mode="human")
    demonstrate(env, q_table)
    env.close()


if __name__ == "__main__":
    main()