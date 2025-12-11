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

## Plotting and Visualisation

The project includes basic tools for visualising how the pollutant concentration changes over time.  
These functions are implemented in `src/plots.py`.

### 1. Animation of the Advection Process

`animate_advection(x, t, C, ...)` creates an animation showing how the concentration profile moves downstream during the simulation.

**Inputs:**
- **x** – spatial grid  
- **t** – time grid  
- **C** – concentration array (nt × nx)  
- **interval** – optional delay between frames  

This is useful for quickly seeing how the pollutant plume evolves.

### 2. Snapshot Plots

`plot_space_time_snapshots(x, t, C, ...)` plots the concentration at several different time steps on the same figure, helping compare how the profile changes throughout the simulation.

These visualisation tools help interpret the results produced by the numerical solver.


## Test Cases

The project includes several test cases to evaluate model behaviour under different numerical and physical conditions.  
These tests help confirm that the advection model behaves as expected and that the numerical setup is robust.

---

### Test Case 3 – Parameter Sensitivity

This test investigates how changes in the key numerical parameters affect the final pollutant distribution:

- Flow velocity **U**
- Spatial resolution **dx**
- Time step **dt**

For multiple combinations of (U, dx, dt), the model is run and:

- The maximum concentration is recorded  
- The location of the concentration peak is tracked  
- Final concentration profiles are plotted to compare sensitivity to each parameter  

An example animation is also produced for one parameter set.  
This test demonstrates stability, resolution effects, and how parameter choices influence plume movement.

---

### Test Case 4 – Exponential Decay at the Inlet

This test applies an exponentially decaying inlet concentration:

$C_{\text{in}}(t) = C_0 e^{-\lambda t}$


Several decay rates **λ** (e.g. 0.0, 0.005, 0.01, 0.02) are tested.  
For each λ:

- The model is run with a decaying source  
- The final plume shape is compared  
- Maximum concentration values are recorded  

Plots show how stronger decay rates reduce downstream concentrations.  
A space–time plot and animation can be generated for selected λ values.

---

### Test Case 5 – Variable Velocity Profile

This test explores the impact of an unsteady flow velocity:

- The base velocity is **U₀ = 0.1 m/s**
- A noisy time series U(t) is generated with ±10% random variation  
- Concentration fields are compared for:
  1. Constant velocity  
  2. Variable velocity U(t)

Outputs include:

- A plot of U(t) vs time  
- A comparison of final concentration profiles (constant vs variable U)  
- A space–time plot of the variable-velocity case  

This test shows how fluctuating flow conditions influence pollutant transport.

---

These test cases collectively evaluate numerical stability, physical behaviour, and parameter sensitivity of the pollutant transport model.





















