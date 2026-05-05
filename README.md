# Minecraft Model Reader

A Python library for reading Minecraft's various resource pack formats (Java and Bedrock).

## Installation

```bash
pip install -e .
```

### Dependency Note: numpy < 2.0

This project depends on [amulet-nbt](https://pypi.org/project/amulet-nbt/), which is compiled (Cython) against numpy 1.x. **numpy >= 2.0 is not supported** — it will cause a binary incompatibility crash at import time:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
```

The `numpy>=1.17,<2.0` constraint in `setup.cfg` enforces this. If you have other packages in your environment that require numpy >= 2.0 (e.g. `ml-dtypes`, `xarray-dataclass`), use a separate virtual environment for this project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```python
from minecraft_model_reader.api import Block, BlockMesh
from minecraft_model_reader.api.resource_pack import (
    load_resource_pack,
    load_resource_pack_manager,
)

# Load the bundled Java vanilla fix resource pack
pack = load_resource_pack("path/to/resource_pack")
manager = load_resource_pack_manager(["path/to/resource_pack"])

# Get a block model
block = Block(namespace="minecraft", base_name="stone")
model = manager.get_block_model(block)

print(model.textures)       # texture paths
print(model.is_transparent) # transparency mode
print(model.faces)          # face data by cull direction
```

## Running Tests

```bash
python tests/test_verify_package.py
```

## License

See [setup.cfg](setup.cfg) for package metadata.
