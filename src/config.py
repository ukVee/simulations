from dataclasses import dataclass
import torch


@dataclass
class SimConfig:
    world_height: int = 64
    world_width: int = 64
    initial_agents: int = 500
    max_ticks: int = 500
    log_interval: int = 50

    hunger_rate: float = 0.5          # hunger increment per tick
    hunger_health_loss: float = 0.2   # health lost per tick
    hunger_threshold: float = 8.0

    food_spawn_prob: float = 0.02     # probability per empty tile per tick
    food_health_gain: float = 5.0

    share_fraction: float = 0.1
    share_min_health: float = 2.0

    reproduction_health_threshold: float = 12.0
    reproduction_mutation_std: float = 0.05

    seed: int = 123

    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def set_seed(self) -> None:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
