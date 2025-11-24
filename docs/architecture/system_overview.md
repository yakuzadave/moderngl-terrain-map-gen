# System Architecture

This document provides a high-level overview of the `map_gen` system architecture, component relationships, and data flow.

## High-Level Architecture

The system is designed as a pipeline where configuration drives procedural generation on the GPU, resulting in data structures that can be exported or visualized.

```mermaid
---
id: 50aa8f8d-6a18-465b-b727-bbe38c9d193c
---
graph TD
    User[User] -->|CLI Args| CLI[gpu_terrain.py]
    User -->|Interactions| UI[app/ui_streamlit.py]
    
    subgraph "Entry Points"
        CLI
        UI
    end
    
    CLI -->|Config| Gen[Generators]
    UI -->|Config| Gen
    
    subgraph "Core Logic (src/generators)"
        Gen --> Erosion[ErosionTerrainGenerator]
        Gen --> Hydro[HydraulicErosionGenerator]
        Gen --> Morph[MorphologicalTerrainGPU]
    end
    
    subgraph "Rendering Engine (ModernGL)"
        Ctx[GL Context]
        Shaders[GLSL Shaders]
        Buffers[Framebuffers/Textures]
        
        Erosion -->|Uses| Ctx
        Erosion -->|Compiles| Shaders
        Erosion -->|Renders to| Buffers
        Hydro -->|Uses| Ctx
    end
    
    subgraph "Data Model"
        Maps[TerrainMaps Object]
    end
    
    Buffers -->|ReadPixels| Maps
    
    subgraph "Utilities (src/utils)"
        Export[Export Utils]
        Viz[Visualization Utils]
        Tex[Texture Utils]
    end
    
    Maps -->|Save PNG/OBJ| Export
    Maps -->|Matplotlib| Viz
    Maps -->|Pack Channels| Tex
    
    Export --> Files[(Output Files)]
```

## Class Structure

The core generation logic is encapsulated in generator classes that manage their own ModernGL resources. Data is passed around using the `TerrainMaps` data transfer object.

```mermaid
classDiagram
    class TerrainMaps {
        +ndarray height
        +ndarray normals
        +ndarray erosion_mask
        +int resolution
        +ensure(data) TerrainMaps
        +height_u16() ndarray
    }

    class ErosionParams {
        +float height_tiles
        +int height_octaves
        +float erosion_strength
        +int thermal_iterations
        +canyon() ErosionParams
        +plains() ErosionParams
        +mountains() ErosionParams
        +uniforms() dict
    }

    class ErosionTerrainGenerator {
        +Context ctx
        +Program height_program
        +Program viz_program
        +Program thermal_program
        +generate_heightmap(seed, params) TerrainMaps
        +render_visualization(mode) ndarray
        +render_raymarch(params) ndarray
        +apply_thermal_erosion(iters)
        +cleanup()
    }

    class HydraulicErosionGenerator {
        +Context ctx
        +Program erosion_program
        +simulate(heightmap, params) TerrainMaps
    }

    class MorphologicalTerrainGPU {
        +generate(resolution, seed) TerrainMaps
    }

    ErosionTerrainGenerator ..> ErosionParams : uses
    ErosionTerrainGenerator ..> TerrainMaps : produces
    HydraulicErosionGenerator ..> TerrainMaps : produces
    MorphologicalTerrainGPU ..> TerrainMaps : produces
```

## GPU Generation Pipeline

The generation process relies heavily on the GPU. Python acts as the orchestrator, setting up the state and retrieving the results.

```mermaid
sequenceDiagram
    participant Py as Python (Generator)
    participant GL as ModernGL Context
    participant VS as Vertex Shader
    participant FS as Fragment Shader (Compute)
    participant FBO as Framebuffer

    Note over Py, FBO: Initialization
    Py->>GL: Create Context (Headless/Window)
    Py->>GL: Compile Shaders (quad.vert + *.frag)
    Py->>GL: Create Textures & FBOs

    Note over Py, FBO: Generation Pass
    Py->>GL: Upload Uniforms (Seed, Octaves, Params)
    Py->>GL: Bind Quad VBO
    Py->>GL: Render()
    
    GL->>VS: Process Vertices (Full-screen Quad)
    VS->>FS: Interpolate UVs
    FS->>FS: Generate Noise (FBM)
    FS->>FS: Apply Domain Warping
    FS->>FS: Apply Erosion Math
    FS->>FBO: Output (Height, Normal, Mask)

    Note over Py, FBO: Thermal Erosion (Optional)
    loop Thermal Iterations
        Py->>GL: Render Thermal Pass (Ping-Pong)
        FS->>FS: Calculate Talus Deposition
        FS->>FBO: Update Heightmap
    end

    Note over Py, FBO: Readback
    Py->>FBO: Read Pixels (f4)
    FBO-->>Py: Raw Bytes
    Py->>Py: Convert to NumPy (TerrainMaps)
```

## Hydraulic Erosion Simulation Loop

The hydraulic erosion generator uses a complex iterative simulation loop to model water flow and sediment transport.

```mermaid
stateDiagram-v2
    [*] --> Init
    Init --> Loop
    
    state Loop {
        Flux: Calculate Water Flux
        Water: Update Water Depth & Velocity
        Erosion: Erode/Deposit Sediment
        Advection: Move Sediment
        Evaporation: Evaporate Water
        Thermal: Thermal Erosion (Slope)
        
        Flux --> Water
        Water --> Erosion
        Erosion --> Advection
        Advection --> Evaporation
        Evaporation --> Thermal
    }
    
    Thermal --> Loop : Iterations < Max
    Thermal --> Finish : Iterations >= Max
    
    Finish --> [*]
```

## Directory Structure

```mermaid
graph LR
    Root[map_gen]
    
    Root --> Src[src/]
    Src --> Gen[generators/]
    Src --> Utils[utils/]
    Src --> Shaders[shaders/]
    
    Gen --> Erosion[erosion.py]
    Gen --> Hydro[hydraulic.py]
    
    Utils --> Export[export.py]
    Utils --> Render[rendering.py]
    Utils --> Artifacts[artifacts.py]
    
    Shaders --> Height[erosion_heightmap.frag]
    Shaders --> Viz[erosion_viz.frag]
    Shaders --> Ray[erosion_raymarch.frag]
    Shaders --> Thermal[thermal_erosion.frag]
```
