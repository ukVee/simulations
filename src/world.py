import torch
from .config import SimConfig


def create_world(cfg: SimConfig) -> dict:
    grid = torch.zeros((cfg.world_height, cfg.world_width), device=cfg.device, dtype=torch.int8)
    return {"grid": grid}


def regenerate_food(world: dict, cfg: SimConfig) -> dict:
    empty = world["grid"] == 0
    spawn = torch.rand_like(world["grid"], dtype=torch.float32) < cfg.food_spawn_prob
    world["grid"] = torch.where(empty & spawn, torch.ones_like(world["grid"]), world["grid"])
    return world
