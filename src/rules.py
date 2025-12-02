import torch
from .config import SimConfig


def movement_step(state: dict, cfg: SimConfig) -> dict:
    directions = torch.tensor(
        [[0, 1], [0, -1], [1, 0], [-1, 0]], device=cfg.device, dtype=torch.int64
    )
    idx = torch.randint(0, 4, (state["x"].shape[0],), device=cfg.device)
    delta = directions[idx]
    state["x"] = torch.clamp(state["x"] + delta[:, 0], 0, cfg.world_width - 1)
    state["y"] = torch.clamp(state["y"] + delta[:, 1], 0, cfg.world_height - 1)
    return state


def hunger_step(state: dict, cfg: SimConfig) -> dict:
    state["hunger"] = state["hunger"] + cfg.hunger_rate
    state["health"] = state["health"] - cfg.hunger_health_loss
    return state


def food_step(state: dict, world: dict, cfg: SimConfig) -> tuple[dict, dict]:
    idx_flat = state["y"] * cfg.world_width + state["x"]
    grid_flat = world["grid"].view(-1)
    has_food = grid_flat[idx_flat] > 0

    # consume
    state["hunger"] = torch.where(has_food, torch.zeros_like(state["hunger"]), state["hunger"])
    state["health"] = torch.where(
        has_food, state["health"] + cfg.food_health_gain, state["health"]
    )

    # remove consumed food
    consumed_indices = idx_flat[has_food]
    if consumed_indices.numel() > 0:
        grid_flat = grid_flat.index_fill(0, consumed_indices, 0)
        world["grid"] = grid_flat.view(world["grid"].shape)

    return state, world


def sharing_step(state: dict, cfg: SimConfig) -> dict:
    altruistic = torch.rand_like(state["altruism"]) < state["altruism"]
    donors = altruistic & (state["health"] > cfg.share_min_health)
    donations = state["health"] * cfg.share_fraction * donors.float()

    state["health"] = state["health"] - donations
    total_donation = donations.sum()

    recipients = state["hunger"] > cfg.hunger_threshold
    rec_count = recipients.sum()
    if rec_count > 0 and total_donation > 0:
        share = total_donation / rec_count
        state["health"][recipients] += share
    return state


def reproduction_step(state: dict, cfg: SimConfig) -> dict:
    parents = state["health"] >= cfg.reproduction_health_threshold
    count = int(parents.sum().item())
    if count == 0:
        return state

    # split energy
    parent_energy = state["health"][parents] * 0.5
    state["health"][parents] = parent_energy

    child_x = state["x"][parents] + torch.randint(-1, 2, (count,), device=cfg.device)
    child_y = state["y"][parents] + torch.randint(-1, 2, (count,), device=cfg.device)
    child_x = torch.clamp(child_x, 0, cfg.world_width - 1)
    child_y = torch.clamp(child_y, 0, cfg.world_height - 1)

    child_health = parent_energy
    child_hunger = state["hunger"][parents] * 0.5
    child_altruism = torch.clamp(
        state["altruism"][parents] + torch.randn(count, device=cfg.device) * cfg.reproduction_mutation_std,
        0.0,
        1.0,
    )

    state["x"] = torch.cat([state["x"], child_x], dim=0)
    state["y"] = torch.cat([state["y"], child_y], dim=0)
    state["health"] = torch.cat([state["health"], child_health], dim=0)
    state["hunger"] = torch.cat([state["hunger"], child_hunger], dim=0)
    state["altruism"] = torch.cat([state["altruism"], child_altruism], dim=0)
    return state


def filter_dead(state: dict) -> dict:
    alive = state["health"] > 0
    for key in state.keys():
        state[key] = state[key][alive]
    return state
