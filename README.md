# CNM_2025_group_04 – River Pollutant Transport Model

This repository contains the group coursework for CNM 2025.  
The task is to simulate the 1D transport of a pollutant in a river using the advection equation:

∂C/∂t + U · ∂C/∂x = 0

The project integrates numerical modelling, Python programming, GitHub collaboration, and scientific reporting.

---

## 📁 Repository Structure

.
├── README.md
├── data/
│   └── initial_conditions.csv
├── results/
├── src/
│   ├── solver.py
│   ├── initial_conditions.py
│   ├── plotting.py
│   └── main.py
└── tests/

## Branch Structure

- main
- feature-initial-conditions
- feature-plots
- feature-readme
- feature-solver
- feature-tests

---

## Initial Conditions

The model can start with two types of initial pollutant distributions:

### 1. Point-source initial condition
- Only the inlet (x = 0) has pollutant at the start  
- Suitable for simple or idealised test cases  

### 2. CSV-based initial condition
- Data is read from `data/initial_conditions.csv`  
- Values are cleaned and interpolated to match the model grid  

These functions in `src/initial_conditions.py` handle the process:
- `read_initial_conditions(path)`  
- `interpolate_to_grid(df, grid_x)`  
- `load_initial_condition_on_grid(path, grid_x)`  

This allows the model to use either simple or realistic pollutant profiles.

--- 


## Numerical Solver

The pollutant transport is simulated using the 1D advection–decay equation:

∂C/∂t + U · ∂C/∂x = −λC

The solver (in `src/solver.py`) uses a backward (implicit) upwind scheme, which is stable for larger time steps and suitable for modelling pollutant movement in a river.

### Key functions
- `create_space_grid(...)` → builds the spatial grid  
- `create_time_grid(...)` → builds the time grid  
- `advect_1d_backward(...)` → runs the advection–decay model  
  - supports constant or varying velocity  
  - supports optional decay  
  - allows time-dependent inlet boundary conditions  

The `run_base_case()` function runs the full simulation using the coursework setup (20 m domain, 300 s, 0.1 m/s velocity, 250 µg/m³ inlet).






