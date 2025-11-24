# Hydraulic Erosion Simulation

This module implements a physical hydraulic erosion simulation based on the **Pipe Model** (or a simplified shallow water / particle-grid hybrid) adapted for OpenGL Fragment Shaders.

## Overview

Unlike the noise-based erosion in the standard generator (which uses domain warping and slope-based subtraction), this generator simulates the flow of water and sediment transport over time.

### Key Features

- **Physical Simulation**: Models water accumulation, velocity, sediment capacity, and deposition.
- **Ping-Pong Rendering**: Uses pairs of framebuffers to simulate time steps on the GPU.
- **Parameters**:
  - `dt`: Time step (smaller = more stable).
  - `iterations`: Number of simulation steps.
  - `rain_rate`: Water added per step.
  - `evaporation_rate`: Water removed per step.
  - `sediment_capacity`: How much sediment water can carry.
  - `deposition/dissolving`: Rates of mass transfer between terrain and water.

## Architecture

The simulation runs in a loop of 5 passes per iteration:

1.  **Flux Pass**: Calculates water outflow from each cell to its neighbors based on height differences.
2.  **Water Pass**: Updates water depth and calculates velocity vectors.
3.  **Erosion/Deposition Pass**: Dissolves terrain into sediment or deposits sediment onto terrain based on velocity and capacity.
4.  **Advection Pass**: Moves suspended sediment along the velocity field (Semi-Lagrangian).
5.  **Evaporation Pass**: Reduces water depth.

## Usage

### Python API

```python
from src import HydraulicErosionGenerator, HydraulicParams
import numpy as np

# 1. Create generator
gen = HydraulicErosionGenerator(resolution=512)

# 2. Prepare initial heightmap (e.g., from noise)
# heightmap should be a float32 numpy array (512x512)
initial_height = ... 

# 3. Configure parameters
params = HydraulicParams(
    iterations=100,
    dt=0.002,
    rain_rate=0.01,
    evaporation_rate=0.01
)

# 4. Run simulation
terrain = gen.simulate(initial_height, params)

# 5. Cleanup
gen.cleanup()
```

### Streamlit UI

1.  Launch the UI: `streamlit run app/ui_streamlit.py`
2.  In the Sidebar, change **Generator Type** to `hydraulic`.
3.  Adjust parameters in the **Terrain > Erosion** tab.
4.  Click **Generate**.

## Implementation Details

- **Shaders**: Located in `src/shaders/hydraulic/`.
- **Class**: `src.generators.hydraulic.HydraulicErosionGenerator`.
- **Stability**: The simulation can be sensitive to `dt`. If you see artifacts or NaNs, try reducing `dt` (e.g., 0.001 or 0.002).
