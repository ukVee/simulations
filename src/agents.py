import torch
from .config import SimConfig


def create_agents(cfg: SimConfig) -> dict:
    n = cfg.initial_agents
    x = torch.randint(0, cfg.world_width, (n,), device=cfg.device, dtype=torch.int64)
    y = torch.randint(0, cfg.world_height, (n,), device=cfg.device, dtype=torch.int64)
    health = torch.full((n,), 10.0, device=cfg.device)
    hunger = torch.zeros((n,), device=cfg.device)
    altruism = torch.clamp(torch.normal(0.3, 0.1, size=(n,), device=cfg.device), 0.0, 1.0)
    return {"x": x, "y": y, "health": health, "hunger": hunger, "altruism": altruism}
