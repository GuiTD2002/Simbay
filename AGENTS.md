# Repository Guidelines

## Project Structure & Module Organization

This is a Python robotics and MuJoCo simulation repository. Core code lives in `src/`, grouped by domain:

- `src/robots/`: real and MuJoCo robot interfaces.
- `src/planning/` and `src/kinematics/`: trajectory generation and Franka IK helpers.
- `src/estimation/`: particle filters, motion models, measurements, and robot containers.
- `src/skills/`: task-level actions such as sweeps, homing, clicking, and lifting.
- `src/utils/`: constants, MuJoCo setup, sensors, and plotting helpers.

Executable experiments live in `scripts/`. MuJoCo XML, STL, and mesh assets are under `models/` and `models/assets/`. Reference plots are stored in `saved_plots/`. `test_robot.py` is an integration-style robot script, not a unit test suite.

## Build, Test, and Development Commands

Run commands from the repository root so `src` imports and relative model paths resolve correctly.

- `python scripts/pos_estimation_1D.py`: run the 1D particle-filter demo in simulation by default.
- `python scripts/pos_estimation_2D.py`: run the 2D position estimation demo.
- `python scripts/mass_estimation.py`: run mass estimation and update plot output.
- `python -m compileall src scripts test_robot.py`: syntax-check project Python files without running robot code.
- `python test_robot.py`: execute the real-robot integration sequence; only run when hardware is connected and safe.

No `requirements.txt`, `pyproject.toml`, or build system is checked in. Install dependencies used by the code, including `numpy`, `matplotlib`, and `mujoco`, locally.

## Coding Style & Naming Conventions

Use Python style with 4-space indentation. Use `snake_case` for functions, variables, and modules; use `PascalCase` for classes such as `RobotContainer` and `ParticleFilterRegularized`. Keep imports grouped as standard library, third-party, then local `src` imports. Prefer small, explicit functions over hidden script state.

## Testing Guidelines

There is no formal test framework configured yet. Before submitting changes, run `python -m compileall src scripts test_robot.py` and at least one relevant simulation script. Avoid running real-robot scripts unless the change specifically targets hardware behavior and the environment is prepared.

## Commit & Pull Request Guidelines

The history only shows `Initial Commit`, so use concise imperative commit messages such as `Add sweep contact guard` or `Fix particle resampling bounds`. Pull requests should include a summary, commands run, affected modules, and generated plot changes. For hardware-facing changes, state whether they were tested in simulation, on the real robot, or not tested.

## Security & Configuration Tips

Do not commit local environments, logs, compiled MuJoCo `.mjb` files, or IDE settings; these are covered by `.gitignore`. Keep robot-specific credentials, IPs, and calibration details out of source files unless they are safe defaults.
