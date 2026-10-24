import json
import random
from pathlib import Path

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibilidad
random.seed(0)
torch.manual_seed(0)

# Configuración del entorno
ZONES = 3
FLEET = 6
HORIZON = 12

# Acción 0: no mover.
# Las demás acciones: mover un vehículo de una zona a otra.
ACTIONS = [None] + [
    (src, dst)
    for src in range(ZONES)
    for dst in range(ZONES)
    if src != dst
]
N_ACTIONS = len(ACTIONS)

# Demanda artificial para cada una de las 12 horas.
DEMAND_PATTERN = [
    [3, 1, 2],
    [3, 2, 1],
    [2, 3, 1],
    [1, 4, 2],
    [1, 4, 3],
    [1, 3, 4],
    [2, 2, 4],
    [3, 1, 3],
    [4, 1, 2],
    [4, 2, 1],
    [3, 3, 1],
    [3, 3, 2],
]

def make_demands(rng):
    """Genera demanda con pequeñas variaciones aleatorias."""
    demands = []

    for base_demand in DEMAND_PATTERN:
        demand = [
            max(0, value + rng.choice([-1, 0, 0, 1]))
            for value in base_demand
        ]
        demands.append(demand)

    return demands

def observe(cars, demand, t):
    """Estado: vehículos por zona, demanda por zona y hora."""
    normalized_cars = [value / FLEET for value in cars]
    normalized_demand = [value / 5.0 for value in demand]
    normalized_time = [t / HORIZON]

    return normalized_cars + normalized_demand + normalized_time

def heuristic_action(cars, demand):
    """Mueve de una zona con vehículos sobrantes a una con demanda pendiente."""
    best_score = 0
    best_action = 0

    for action_id, move in enumerate(ACTIONS):
        if move is None:
            continue

        src, dst = move
        surplus = cars[src] - demand[src]
        shortage = demand[dst] - cars[dst]

        if surplus > 0 and shortage > 0:
            score = min(surplus, shortage)

            if score > best_score:
                best_score = score
                best_action = action_id

    return best_action

def transition(cars, demand, action):
    """Ejecuta una acción y devuelve la recompensa y los viajes atendidos."""
    moved = False

    if action != 0:
        src, dst = ACTIONS[action]

        if cars[src] > 0:
            cars[src] -= 1
            cars[dst] += 1
            moved = True

    rides = sum(
        min(cars[zone], demand[zone])
        for zone in range(ZONES)
    )

    move_cost = 0.25 if moved else 0.0
    reward = rides - move_cost
    return reward, rides

def make_episode(seed):
    """Genera una trayectoria sintética para entrenar el modelo."""
    rng = random.Random(seed)
    demand_tape = make_demands(rng)
    cars = [2, 2, 2]

    states = []
    actions = []
    rewards = []

    for t in range(HORIZON):
        demand = demand_tape[t]
        states.append(observe(cars, demand, t))

        # La mayoría de acciones vienen de una heurística.
        # Algunas son aleatorias para agregar variedad a los datos.
        if rng.random() < 0.15:
            action = rng.randrange(N_ACTIONS)
        else:
            action = heuristic_action(cars, demand)

        reward, _ = transition(cars, demand, action)

        actions.append(action)
        rewards.append(reward)

    # Retorno futuro normalizado.
    returns_to_go = [0.0] * HORIZON
    running_return = 0.0

    for t in reversed(range(HORIZON)):
        running_return += rewards[t]
        returns_to_go[t] = running_return / HORIZON

    return {
        "states": states,
        "actions": actions,
        "rtg": returns_to_go,
        "rewards": rewards,
    }

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, d_model=64, n_heads=4, n_layers=2):
        super().__init__()

        self.state_embed = nn.Linear(state_dim, d_model)
        self.return_embed = nn.Linear(1, d_model)

        # El índice N_ACTIONS se usa como token de inicio.
        self.action_embed = nn.Embedding(N_ACTIONS + 1, d_model)
        self.position_embed = nn.Embedding(HORIZON, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
nhead=n_heads,
            batch_first=True,
            dropout=0.1,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.action_head = nn.Linear(d_model, N_ACTIONS)

    def forward(self, states, returns_to_go, previous_actions):
        batch_size, length, _ = states.shape
        positions = torch.arange(length, device=states.device)

        # Cada paso combina estado, retorno deseado y acción anterior.
        x = (
            self.state_embed(states)
            + self.return_embed(returns_to_go.unsqueeze(-1))
            + self.action_embed(previous_actions)
            + self.position_embed(positions)[None, :, :]
        )

        # Máscara causal: el modelo no puede ver pasos futuros.
        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=states.device,
            ),
            diagonal=1,
        )

        x = self.transformer(x, mask=causal_mask)
        return self.action_head(x)

def train_model():
    print("Generando trayectorias...")

    episodes = [
        make_episode(seed=1000 + i)
        for i in range(400)
    ]

    states = torch.tensor(
        [episode["states"] for episode in episodes],
        dtype=torch.float32,
    )
    actions = torch.tensor(
        [episode["actions"] for episode in episodes],
        dtype=torch.long,
    )
    returns_to_go = torch.tensor(
        [episode["rtg"] for episode in episodes],
        dtype=torch.float32,
    )

    model = DecisionTransformer(state_dim=2 * ZONES + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    batch_size = 32
    training_steps = 300

    print("Entrenando Decision Transformer...")

    for step in range(training_steps):
        indices = torch.randint(len(episodes), (batch_size,))

        batch_states = states[indices]
        batch_actions = actions[indices]
        batch_returns = returns_to_go[indices]

        # En el paso t, el modelo ve la acción del paso t-1.
        previous_actions = torch.full_like(batch_actions, N_ACTIONS)
        previous_actions[:, 1:] = batch_actions[:, :-1]

        logits = model(
            batch_states,
            batch_returns,
            previous_actions,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, N_ACTIONS),
            batch_actions.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    episode_returns = [
        sum(episode["rewards"])
        for episode in episodes
    ]
    episode_returns.sort()

    # Condicionamos el modelo a un retorno alto, pero no extremo.
    percentile_index = int(0.9 * (len(episode_returns) - 1))
    target_return = episode_returns[percentile_index]

    print(f"Pérdida final: {loss.item():.3f}")
    print(f"Retorno deseado: {target_return:.2f}")

    return model, target_return

def run_episode(
    seed,
    policy,
    model=None,
    target_return=0.0,
    return_history=False,
):
    """Ejecuta un episodio y, opcionalmente, guarda datos para graficar."""
    rng = random.Random(seed)
    demand_tape = make_demands(rng)
    cars = [2, 2, 2]

    total_reward = 0.0
    total_rides = 0
    remaining_return = target_return
    previous_action = N_ACTIONS

    state_history = []
    return_history_values = []
    action_history = []

    history = {
        "cars": [],
        "demand": [],
        "actions": [],
        "rewards": [],
    }

    for t in range(HORIZON):
        demand = demand_tape[t]

        if policy == "dt":
            state_history.append(observe(cars, demand, t))
            return_history_values.append(remaining_return / HORIZON)
            action_history.append(previous_action)

            with torch.no_grad():
                logits = model(
                    torch.tensor(
                        [state_history],
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        [return_history_values],
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        [action_history],
                        dtype=torch.long,
                    ),
                )

                action = int(logits[0, -1].argmax())

        elif policy == "heuristic":
            action = heuristic_action(cars, demand)

        else:
            # Política de referencia: nunca mueve vehículos.
            action = 0

        reward, rides = transition(cars, demand, action)

        total_reward += reward
        total_rides += rides
        remaining_return -= reward
        previous_action = action

        # Se guardan los vehículos después de la posible reubicación.
        history["cars"].append(cars.copy())
        history["demand"].append(demand.copy())
        history["actions"].append(action)
        history["rewards"].append(reward)

    if return_history:
        return total_reward, total_rides, history

    return total_reward, total_rides

def export_episode(history, seed, target_return):
    """Guarda un episodio para reproducirlo en la visualización web."""
    output_path = Path(__file__).with_name("simulation_data.json")
    episode_data = {
        "agent": "Decision Transformer",
        "seed": seed,
        "target_return": target_return,
        "initial_cars": [FLEET // ZONES] * ZONES,
        "action_moves": ACTIONS,
        "history": history,
    }
    output_path.write_text(
        json.dumps(episode_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Datos para la visualización web: {output_path}")

def evaluate(model, target_return):
    """Compara el modelo con dos políticas de referencia."""
    results = {
        "no mover": [],
        "heurística": [],
        "transformer": [],
    }

    policies = [
        ("no mover", "stay"),
        ("heurística", "heuristic"),
        ("transformer", "dt"),
    ]

    # Cada política recibe la misma demanda para cada seed.
    for seed in range(2000, 2050):
        for name, policy in policies:
            result = run_episode(
                seed=seed,
                policy=policy,
                model=model,
                target_return=target_return,
            )
            results[name].append(result)

    print("\nPromedio en 50 episodios:")

    for name, scores in results.items():
        avg_reward = sum(score[0] for score in scores) / len(scores)
        avg_rides = sum(score[1] for score in scores) / len(scores)

        print(
            f"{name:12s} "
            f"recompensa={avg_reward:5.2f}, "
            f"viajes={avg_rides:5.2f}"
        )

def plot_episode(history):
    """Anima las métricas del episodio y los vehículos sobre una cuadrícula."""
    if not history["cars"]:
        raise ValueError("No hay pasos en el historial para animar.")

    zones = list(range(ZONES))
    width = 0.35
    fig, (bars_ax, grid_ax) = plt.subplots(1, 2, figsize=(13, 6))
    car_bars = bars_ax.bar(
        [zone - width / 2 for zone in zones],
        history["cars"][0],
        width,
        label="Vehículos",
        color="tab:blue",
    )
    demand_bars = bars_ax.bar(
        [zone + width / 2 for zone in zones],
        history["demand"][0],
        width,
        label="Demanda",
        color="tab:orange",
    )
    status = fig.suptitle("")

    grid_size = 5
    zone_colors = ["#eaf3ff", "#eaf8ef", "#fff3e5"]
    vehicle_slots = [
        (0.8, 0.8),
        (2.5, 0.8),
        (4.2, 0.8),
        (0.8, 2.5),
        (2.5, 2.5),
        (4.2, 2.5),
    ]
    for zone in zones:
        left = zone * grid_size
        grid_ax.axvspan(
            left,
            left + grid_size,
            facecolor=zone_colors[zone],
            zorder=0,
        )
        grid_ax.text(
            left + grid_size / 2,
            grid_size + 0.2,
            f"Zona {zone + 1}\nDemanda: {history['demand'][0][zone]}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    grid_ax.set_xlim(0, ZONES * grid_size)
    grid_ax.set_ylim(0, grid_size + 0.9)
    grid_ax.set_xticks(range(ZONES * grid_size + 1), minor=True)
    grid_ax.set_yticks(range(grid_size + 1), minor=True)
    grid_ax.grid(which="minor", color="gray", linewidth=0.7, alpha=0.7)
    grid_ax.set_xticks([])
    grid_ax.set_yticks([])
    grid_ax.tick_params(which="minor", bottom=False, left=False)
    grid_ax.set_title("Mapa de vehículos (cada cuadro = una celda)")
    grid_ax.set_aspect("equal")

    vehicles = grid_ax.scatter(
        [],
        [],
        s=180,
        color="tab:blue",
        edgecolors="white",
        linewidths=1,
        zorder=3,
        label="Vehículo",
    )
    grid_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=1)
    action_annotation = [None]

    bars_ax.set_xticks(zones, [f"Zona {zone + 1}" for zone in zones])
    bars_ax.set_ylim(
        0,
        max(FLEET, max(max(row) for row in history["demand"])) + 1,
    )
    bars_ax.set_ylabel("Cantidad")
    bars_ax.set_title("Vehículos y demanda")
    bars_ax.legend()
    bars_ax.grid(axis="y", alpha=0.25)

    def update(frame):
        for bars, values in (
            (car_bars, history["cars"][frame]),
            (demand_bars, history["demand"][frame]),
        ):
            for bar, value in zip(bars, values):
                bar.set_height(value)

        vehicle_positions = []
        for zone, count in enumerate(history["cars"][frame]):
            left = zone * grid_size
            vehicle_positions.extend(
                (left + x, y) for x, y in vehicle_slots[:count]
            )
        vehicles.set_offsets(vehicle_positions)

        for zone, demand in enumerate(history["demand"][frame]):
            grid_ax.texts[zone].set_text(
                f"Zona {zone + 1}\nDemanda: {demand}"
            )

        if action_annotation[0] is not None:
            action_annotation[0].remove()
            action_annotation[0] = None

        action = history["actions"][frame]
        if action == 0:
            action_text = "Sin movimiento"
        else:
            src, dst = ACTIONS[action]
            action_text = f"Z{src + 1} → Z{dst + 1}"
            action_annotation[0] = grid_ax.annotate(
                "",
                xy=(dst * grid_size + grid_size / 2, grid_size / 2),
                xytext=(src * grid_size + grid_size / 2, grid_size / 2),
                arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 2},
            )

        status.set_text(
            f"Paso {frame + 1}/{len(history['cars'])} | "
            f"Acción: {action_text} | "
            f"Recompensa: {history['rewards'][frame]:.2f}"
        )
        return (*car_bars, *demand_bars, status)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history["cars"]),
        interval=700,
        repeat=False,
        blit=False,
    )
    fig._episode_animation = animation
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, target = train_model()
    model.eval()

    evaluate(model, target)

    # Ejecuta y muestra un episodio del Decision Transformer.
    _, _, episode_history = run_episode(
        seed=42,
        policy="dt",
        model=model,
        target_return=target,
        return_history=True,
    )

    export_episode(episode_history, seed=42, target_return=target)
    plot_episode(episode_history)