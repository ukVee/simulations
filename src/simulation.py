from typing import Tuple
from .config import SimConfig
from . import world as world_mod
from . import agents
from . import rules


def run_simulation(cfg: SimConfig) -> Tuple[dict, dict]:
    cfg.set_seed()
    world = world_mod.create_world(cfg)
    state = agents.create_agents(cfg)

    for tick in range(cfg.max_ticks):
        world = world_mod.regenerate_food(world, cfg)
        state = rules.movement_step(state, cfg)
        state = rules.hunger_step(state, cfg)
        state, world = rules.food_step(state, world, cfg)
        state = rules.sharing_step(state, cfg)
        state = rules.reproduction_step(state, cfg)
        state = rules.filter_dead(state)

        if (tick + 1) % cfg.log_interval == 0:
            print(f"[tick {tick+1}] population={state['health'].shape[0]}")
    return state, world
