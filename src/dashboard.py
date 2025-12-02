"""Mesa Solara dashboard wrapping the PyTorch GPU simulation.

Run with: `solara run src/dashboard.py --host 0.0.0.0 --port 8765`
"""

from __future__ import annotations

import torch
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.visualization import SolaraViz, make_plot_component, make_space_component

from .config import SimConfig
from . import world as world_mod
from . import agents as agents_mod
from . import rules


class CivAgent(Agent):
    """Lightweight Agent wrapper for visualization only."""

    def __init__(self, unique_id: int, model: Model, x: int, y: int, health: float, hunger: float, altruism: float):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.health = health
        self.hunger = hunger
        self.altruism = altruism

    @property
    def pos(self):
        return (self.x, self.y)


class CivModel(Model):
    def __init__(self, cfg: SimConfig | None = None):
        super().__init__()
        self.cfg = cfg or SimConfig()
        self.cfg.set_seed()

        self.world = world_mod.create_world(self.cfg)
        self.state = agents_mod.create_agents(self.cfg)

        self.grid = MultiGrid(self.cfg.world_width, self.cfg.world_height, torus=False)
        self.agents_list: list[CivAgent] = []

        self.datacollector = DataCollector(
            {
                "population": lambda m: len(m.agents_list),
                "mean_health": lambda m: float(torch.mean(m.state["health"]).item()) if m.state["health"].numel() > 0 else 0.0,
            }
        )

        self._sync_agents_from_state()
        self.datacollector.collect(self)
        self.running = True

    def _sync_agents_from_state(self) -> None:
        """Refresh the Mesa grid/agent objects from the current tensor state (CPU copy for viz only)."""
        self.grid = MultiGrid(self.cfg.world_width, self.cfg.world_height, torus=False)
        self.agents_list = []

        if self.state["x"].numel() == 0:
            return

        x = self.state["x"].detach().to("cpu")
        y = self.state["y"].detach().to("cpu")
        health = self.state["health"].detach().to("cpu")
        hunger = self.state["hunger"].detach().to("cpu")
        altruism = self.state["altruism"].detach().to("cpu")

        for i in range(x.shape[0]):
            agent = CivAgent(
                i,
                self,
                int(x[i].item()),
                int(y[i].item()),
                float(health[i].item()),
                float(hunger[i].item()),
                float(altruism[i].item()),
            )
            self.agents_list.append(agent)
            self.grid.place_agent(agent, (agent.x, agent.y))

    def step(self) -> None:
        # GPU simulation step
        self.world = world_mod.regenerate_food(self.world, self.cfg)
        self.state = rules.movement_step(self.state, self.cfg)
        self.state = rules.hunger_step(self.state, self.cfg)
        self.state, self.world = rules.food_step(self.state, self.world, self.cfg)
        self.state = rules.sharing_step(self.state, self.cfg)
        self.state = rules.reproduction_step(self.state, self.cfg)
        self.state = rules.filter_dead(self.state)

        # Update visualization artifacts
        self._sync_agents_from_state()
        self.datacollector.collect(self)

        if len(self.agents_list) == 0:
            self.running = False


def agent_portrayal(agent: CivAgent):
    """Return a dict portrayal compatible with Mesa's SpaceRenderer."""
    health_norm = max(0.0, min(1.0, agent.health / 15.0))
    r = int((1.0 - health_norm) * 255)
    g = int(health_norm * 255)
    color = f"rgb({r},{g},80)"
    return {"color": color, "size": 6, "marker": "o", "alpha": 0.9}


space_component = make_space_component(agent_portrayal, grid_name="grid")
plot_component = make_plot_component({"Population": "population", "Mean Health": "mean_health"})

page = SolaraViz(
    CivModel(),
    components=[space_component, plot_component],
    name="Civ Sim Dashboard",
)


__all__ = ["CivModel", "CivAgent", "page"]
