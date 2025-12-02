from .config import SimConfig
from .simulation import run_simulation


def main():
    cfg = SimConfig()
    final_state, _ = run_simulation(cfg)
    print(f"Simulation complete. Final population: {final_state['health'].shape[0]}")


if __name__ == "__main__":
    main()
