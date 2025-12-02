# civ-sim

A small, GPU-accelerated, rule-based civilization toy simulation in PyTorch.

## Build

```bash
docker build -t civ-sim -f docker/Dockerfile .
```

## Run (GPU)

```bash
docker run --rm --gpus all civ-sim
```

The `--gpus all` flag exposes all available NVIDIA GPUs to the container; Docker integrates this flag natively when the NVIDIA Container Toolkit is installed. Expect periodic console lines such as `[tick 50] population=487` and a final population count.

### Run the Mesa web dashboard (Solara)

```bash
docker run --rm --gpus all -p 8765:8765 civ-sim solara run src/dashboard.py --host 0.0.0.0 --port 8765
```

Then open http://localhost:8765 in a browser to see the grid and live population/health plots. The dashboard wraps the same GPU simulation; visualization uses CPU copies each tick.

## Project Layout

- `src/config.py` — simulation parameters and device selection.
- `src/world.py` — world grid on CUDA and food regeneration.
- `src/agents.py` — initial agent tensor creation on CUDA.
- `src/rules.py` — vectorized movement, hunger, food gathering, altruism-based sharing, reproduction, and death filtering.
- `src/simulation.py` — orchestrates the tick loop.
- `src/main.py` — entrypoint used by Dockerfile.
- `src/dashboard.py` — Mesa SolaraViz dashboard hooking into the GPU simulation for live visualization.

## Extending

- Add new scalar or vector attributes by extending `agents.create_agents` and threading the tensors through rule functions.
- Introduce new rules in `rules.py` and call them from `simulation.py` to keep the loop modular.
- Tweak parameters in `config.py` to change world size, reproduction thresholds, or food dynamics without touching rule code.

## Notes

- All tensors are created on `device="cuda"` when available and remain device-consistent through vectorized operations.
- The base image already includes CUDA-enabled PyTorch; `requirements.txt` is minimal to avoid overwriting that stack.
