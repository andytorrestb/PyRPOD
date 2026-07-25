# Project Overview

This repository contains a set of Python modules for mission planning, vehicle simulations, plume modeling, and utility functions for aerospace research and development. The modules are organized into directories based on functionality and purpose.

## Directory Structure

### Root Files
- **README.md**: Provides an overview of the project structure and usage instructions.
- **config_test.py**: Script for testing configuration setups.
- **pyrpod.txt**: Text file with core project information.

### Core Modules

#### `mdao/`
- **SweepConfig.py**: Configures parameter sweeps for multi-disciplinary analysis.
- **TradeStudy.py**: Facilitates trade studies for design optimization.

#### `mission/`
- **MissionPlanner.py**: Handles mission planning and execution.

#### `plume/`
- **IsentropicExpansion.py**: Calculates properties of isentropic expansions.
- **RarefiedPlumeGasKinetics.py**: Models the kinetics of rarefied plume gases.

#### `rpod/`
- **JetFiringHistory.py**: Manages and analyzes jet firing data.
- **RPOD.py**: Core logic for rendezvous and proximity operations.
- **header.py**: Contains shared constants and configurations for the `rpod` module.

#### `util/`
- **io/file_print.py**: Utility for formatted output and file handling.
- **stl/transform_stl.json**: JSON configuration for STL transformations.
- **stl/transform_stl.py**: Script for transforming STL files (scaling, translation).

#### `vehicle/`
- **LogisticsModule.py**: Handles logistics of vehicle operations.
- **TargetVehicle.py**: Defines the behavior and characteristics of target vehicles.
- **Vehicle.py**: Base class for generic vehicle simulations.
- **VisitingVehicle.py**: Defines visiting vehicle behavior.

## Usage

### Running the Scripts
Scripts are designed to perform specific tasks such as mission planning, vehicle modeling, or plume analysis. Use appropriate test cases or configurations to execute them.

### Transforming STL Files
To apply transformations to STL files:
1. Edit the `util/stl/transform_stl.json` configuration file with desired scaling and translation parameters.
2. Run the transformation script:
   ```bash
   python util/stl/transform_stl.py util/stl/transform_stl.json
   ```

### Testing
Run tests using the `config_test.py` or dedicated test cases provided for each module.

## Logging

PyRPOD has a centralized, opt-in operational logging system. **Importing PyRPOD
has no logging side effects** — modules never configure handlers on import.

- In a module, get a logger the standard way:
  - `import logging`
  - `logger = logging.getLogger(__name__)`
- At the application boundary, turn logging on explicitly:
  - `from pyrpod.logging_utils import configure_logging`
  - `session = configure_logging(case_dir)`
- Override level/format via env vars (PowerShell):
  - `$env:PYRPOD_LOG_LEVEL = "DEBUG"`
  - `$env:PYRPOD_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"`

Runtime logs are written to `<case_dir>/results/logs/<case-name>_<timestamp>.log`.
See [`docs/logging.md`](../docs/logging.md) for the full architecture, the
`logging.ini` schema, configuration precedence, input checksums, configuration
snapshots, performance/memory caveats, and serial-fallback behavior.

Example:

```python
from pyrpod.logging_utils import configure_logging

session = configure_logging(case_dir)
try:
    # ... run the PyRPOD workflow, e.g. study.jfh_plume_strikes() ...
    session.finalize("successful")
finally:
    session.close()
```

## Adding New Modules
1. Place new scripts in the relevant directory.
2. Ensure the module is well-documented with clear docstrings.
3. Add test cases to validate the new functionality.
4. Follow naming conventions and ensure compatibility with existing modules.
