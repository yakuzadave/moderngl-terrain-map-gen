# Documentation Map

Visual guide to navigating the GPU Terrain Generator documentation.

```
GPU Terrain Generator Documentation
│
├─ 🚀 Getting Started
│  ├─ README.md (Project root) ..................... Quick start & features overview
│  └─ docs/quick-reference.md ...................... Common code patterns & examples
│
├─ 📖 API Documentation
│  ├─ docs/api-reference.md ........................ Complete Python API reference
│  ├─ docs/module-reference.md ..................... Module structure & organization
│  └─ docs/API_DOCUMENTATION_SUMMARY.md ............ Documentation overview
│
├─ 🎯 Feature Guides
│  ├─ docs/HYDRAULIC_EROSION.md .................... Physical erosion simulation
│  ├─ docs/ADVANCED_RENDERING.md ................... Turntables & lighting studies
│  ├─ docs/TEXTURE_EXPORTS.md ...................... Game engine texture formats
│  └─ docs/BATCH_GENERATION.md ..................... Automated generation workflows
│
├─ 💾 Export Documentation
│  ├─ docs/EXPORT_FORMATS.md ....................... Supported file formats
│  ├─ docs/EXPORT_CLI_REFERENCE.md ................. Command line export options
│  ├─ docs/EXPORT_QUICKSTART.md .................... Getting started with exports
│  └─ docs/EXPORT_FEATURE_SUMMARY.md ............... Export capabilities overview
│
├─ ⚙️ Configuration & Setup
│  ├─ docs/config_ui.md ............................ Configuration system guide
│  └─ requirements.txt ............................. Python dependencies
│
└─ 🏗️ Architecture & Design
   ├─ docs/architecture/system_overview.md ......... High-level system design
   ├─ docs/architecture/rendering_pipeline.md ...... Rendering pipeline details
   ├─ docs/architecture/patterns_and_decisions.md .. Design patterns & ADRs
   ├─ docs/architecture/dependency_graph.md ........ Module dependencies
   └─ docs/architecture/knowledge_graph.md ......... Codebase knowledge graph
```

## Documentation by Use Case

### 🎯 "I want to generate terrain quickly"
1. Start: [README.md](../README.md) - Installation & quick start
2. Then: [Quick Reference](quick-reference.md) - Basic terrain generation
3. Advanced: [API Reference](api-reference.md) - Full generator options

### 💾 "I need to export terrain to my game engine"
1. Start: [Export Quickstart](EXPORT_QUICKSTART.md)
2. Reference: [Export Formats](EXPORT_FORMATS.md) - All supported formats
3. CLI: [Export CLI Reference](EXPORT_CLI_REFERENCE.md) - Command line usage
4. API: [API Reference](api-reference.md) → Export Utilities section

### 🎨 "I want to create visualizations"
1. Guide: [Advanced Rendering](ADVANCED_RENDERING.md) - Features overview
2. Examples: [Quick Reference](quick-reference.md) → Rendering section
3. API: [API Reference](api-reference.md) → Rendering & Advanced Rendering sections

### 🌊 "I want realistic erosion"
1. Guide: [Hydraulic Erosion](HYDRAULIC_EROSION.md) - Simulation details
2. Examples: [Quick Reference](quick-reference.md) → Hydraulic Erosion section
3. API: [API Reference](api-reference.md) → HydraulicErosionGenerator

### 🏭 "I need to generate many terrains"
1. Guide: [Batch Generation](BATCH_GENERATION.md) - Workflows
2. Examples: [Quick Reference](quick-reference.md) → Batch Processing section
3. API: [API Reference](api-reference.md) → Batch Processing section

### 🎮 "I'm integrating with Unity/Unreal"
1. Textures: [Texture Exports](TEXTURE_EXPORTS.md) - Splatmaps, AO, packed textures
2. Formats: [Export Formats](EXPORT_FORMATS.md) - Heightmap formats
3. API: [API Reference](api-reference.md) → Texture Utilities section

### 🔧 "I want to extend the codebase"
1. Structure: [Module Reference](module-reference.md) - Codebase organization
2. Design: [Architecture](architecture/) - System design & patterns
3. API: [API Reference](api-reference.md) - Public interfaces

### 🐛 "I'm debugging an issue"
1. Troubleshooting: [Quick Reference](quick-reference.md) → Troubleshooting section
2. Architecture: [System Overview](architecture/system_overview.md)
3. Dependencies: [Dependency Graph](architecture/dependency_graph.md)

## Documentation Depth Levels

### Level 1: Quick Start (5 minutes)
- [README.md](../README.md)
- [Quick Reference](quick-reference.md) → Quick Start section

### Level 2: Common Tasks (15 minutes)
- [Quick Reference](quick-reference.md) - Full read
- [Export Quickstart](EXPORT_QUICKSTART.md)

### Level 3: Feature Deep Dive (30 minutes)
- [Hydraulic Erosion](HYDRAULIC_EROSION.md)
- [Advanced Rendering](ADVANCED_RENDERING.md)
- [Texture Exports](TEXTURE_EXPORTS.md)
- [Batch Generation](BATCH_GENERATION.md)

### Level 4: Complete API (60 minutes)
- [API Reference](api-reference.md) - Complete read
- [Module Reference](module-reference.md)
- [Export Formats](EXPORT_FORMATS.md)

### Level 5: Architecture & Design (90+ minutes)
- [System Overview](architecture/system_overview.md)
- [Rendering Pipeline](architecture/rendering_pipeline.md)
- [Design Patterns](architecture/patterns_and_decisions.md)
- [Module Reference](module-reference.md) - Deep dive

## Quick Navigation by Topic

### Generators
- **API:** [API Reference → Generators](api-reference.md#generators)
- **Module:** [Module Reference → Generators](module-reference.md#generators-package-srcgenerators)
- **Examples:** [Quick Reference → Basic Generation](quick-reference.md#basic-terrain-generation)

### Export & I/O
- **Formats:** [Export Formats](EXPORT_FORMATS.md)
- **API:** [API Reference → Export Utilities](api-reference.md#export-utilities)
- **CLI:** [Export CLI Reference](EXPORT_CLI_REFERENCE.md)
- **Quick Start:** [Export Quickstart](EXPORT_QUICKSTART.md)

### Rendering & Visualization
- **Features:** [Advanced Rendering](ADVANCED_RENDERING.md)
- **API:** [API Reference → Rendering](api-reference.md#rendering-utilities)
- **Examples:** [Quick Reference → Rendering](quick-reference.md#rendering)

### Textures for Game Engines
- **Guide:** [Texture Exports](TEXTURE_EXPORTS.md)
- **API:** [API Reference → Texture Utilities](api-reference.md#texture-utilities)
- **Examples:** [Quick Reference → Textures](quick-reference.md#textures-for-game-engines)

### Configuration
- **Guide:** [Configuration UI](config_ui.md)
- **API:** [API Reference → Configuration](api-reference.md#configuration)
- **Examples:** [Quick Reference → Configuration](quick-reference.md#configuration-files)

### Batch Processing
- **Guide:** [Batch Generation](BATCH_GENERATION.md)
- **API:** [API Reference → Batch Processing](api-reference.md#batch-processing)
- **Examples:** [Quick Reference → Batch Generation](quick-reference.md#batch-generation)

## Documentation Maintenance

### When Adding New Features
1. Update [API Reference](api-reference.md) with new functions
2. Add examples to [Quick Reference](quick-reference.md)
3. Update [Module Reference](module-reference.md) if adding modules
4. Create feature guide in `docs/` if major feature
5. Update this map with new documentation

### When Changing Behavior
1. Update affected sections in [API Reference](api-reference.md)
2. Update examples in [Quick Reference](quick-reference.md)
3. Note in [CHANGELOG.md](../CHANGELOG.md)
4. Update architecture docs if design changes

### When Deprecating Features
1. Mark as deprecated in [API Reference](api-reference.md)
2. Update [Quick Reference](quick-reference.md) with alternatives
3. Document in [CHANGELOG.md](../CHANGELOG.md)
4. Update [Module Reference](module-reference.md)

## Documentation Statistics

**Total Documentation Files:** 20+ Markdown files
**Total Content:** ~100,000+ characters
**API Functions Documented:** 50+ public APIs
**Code Examples:** 100+ working examples
**Guides:** 8+ feature guides
**Architecture Docs:** 5+ design documents

## Search & Discovery

### By File Type
```bash
# API documentation
docs/api-reference.md
docs/module-reference.md
docs/quick-reference.md

# Feature guides
docs/HYDRAULIC_EROSION.md
docs/ADVANCED_RENDERING.md
docs/TEXTURE_EXPORTS.md
docs/BATCH_GENERATION.md

# Export documentation
docs/EXPORT_*.md

# Architecture
docs/architecture/*.md
```

### By Keyword

**"Generator"** → API Reference, Module Reference, Quick Reference
**"Export"** → Export Formats, Export CLI Reference, API Reference
**"Rendering"** → Advanced Rendering, API Reference, Quick Reference
**"Texture"** → Texture Exports, API Reference, Quick Reference
**"Batch"** → Batch Generation, API Reference, Quick Reference
**"Config"** → Configuration UI, API Reference, Quick Reference
**"Hydraulic"** → Hydraulic Erosion, API Reference, Quick Reference

## External Resources

- **GitHub Repository:** Main project repository
- **Requirements:** [requirements.txt](../requirements.txt)
- **Examples:** [examples/](../examples/) directory
- **Tests:** [tests/](../tests/) directory

## Need Help?

1. **Check Quick Reference first:** Most common tasks have examples
2. **Search API Reference:** Full documentation of all functions
3. **Review Feature Guides:** In-depth explanations of major features
4. **Explore Examples:** Working code in `examples/` directory
5. **Check Architecture:** Understanding system design helps debugging

## Documentation Index

For a complete list of all documentation, see:
**[Documentation Index](index.md)**

---

*Documentation Map maintained as part of the GPU Terrain Generator project.*
*Last updated: November 24, 2025*
