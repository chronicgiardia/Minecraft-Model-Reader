"""Test script to load sample Minecraft block models using the library's entry points.

Usage:
    python tests/test_load_sample_model.py

Prerequisites:
    pip install -e .  (install the package in editable mode)

What this script tests:
    1. load_resource_pack()        — auto-detects the bundled Java vanilla fix pack
    2. load_resource_pack_manager() — creates a JavaResourcePackManager from a path
    3. get_block_model()           — loads mesh data (verts, faces, textures, transparency)
                                     for several blocks: stone, dirt, oak_log, oak_stairs, glass
    4. Missing block fallback      — confirms unknown blocks return the missing_block placeholder

Note:
    The bundled java_vanilla_fix pack provides blockstate/model definitions but not
    full vanilla textures, so all blocks resolve to missing_no.png. To get real
    textures, layer a full Minecraft resource pack underneath (see java_resource_pack_test.py).
"""

from minecraft_model_reader.api import Block, BlockMesh
from minecraft_model_reader.api.resource_pack import (
    load_resource_pack,
    load_resource_pack_manager,
)
from minecraft_model_reader.api.resource_pack.java import (
    JavaResourcePack,
    JavaResourcePackManager,
)

import os

# Path to the bundled Java vanilla fix resource pack
VANILLA_FIX_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "minecraft_model_reader",
    "api",
    "resource_pack",
    "java",
    "java_vanilla_fix",
)


def test_load_resource_pack():
    """Test that load_resource_pack auto-detects the Java pack."""
    pack = load_resource_pack(VANILLA_FIX_PATH)
    assert isinstance(pack, JavaResourcePack), f"Expected JavaResourcePack, got {type(pack)}"
    assert pack.valid_pack, "Pack should be valid"
    print(f"[OK] Loaded pack: {pack}")
    print(f"     Format: {pack.pack_format}, Description: {pack.pack_description!r}")


def test_load_resource_pack_manager():
    """Test creating a resource pack manager from a path string."""
    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    assert isinstance(manager, JavaResourcePackManager)
    print(f"[OK] Created resource pack manager with packs: {manager.pack_paths}")
    return manager


def test_get_block_models(manager: JavaResourcePackManager):
    """Load several block models and inspect their mesh data."""
    blocks = {
        "stone": Block(namespace="minecraft", base_name="stone"),
        "dirt": Block(namespace="minecraft", base_name="dirt"),
        "oak_log": Block(
            namespace="minecraft",
            base_name="oak_log",
            properties={"axis": __import__("amulet_nbt").TAG_String("y")},
        ),
        "oak_stairs": Block(
            namespace="minecraft",
            base_name="oak_stairs",
            properties={
                "facing": __import__("amulet_nbt").TAG_String("north"),
                "half": __import__("amulet_nbt").TAG_String("bottom"),
                "shape": __import__("amulet_nbt").TAG_String("straight"),
                "waterlogged": __import__("amulet_nbt").TAG_String("false"),
            },
        ),
        "glass": Block(namespace="minecraft", base_name="glass"),
    }

    for name, block in blocks.items():
        model: BlockMesh = manager.get_block_model(block)
        print(f"\n--- {name} ({block.blockstate}) ---")
        print(f"  Textures:     {model.textures}")
        print(f"  Transparency: {model.is_transparent.name} (opaque={model.is_opaque})")
        print(f"  Face mode:    {model.face_mode}")
        print(f"  Cull dirs:    {list(model.faces.keys())}")
        for direction, faces in model.faces.items():
            n_faces = faces.shape[0] // model.face_mode
            n_verts = model.verts[direction].shape[0] // 3
            print(f"    {str(direction):>5s}: {n_faces} faces, {n_verts} verts")


def test_missing_block(manager: JavaResourcePackManager):
    """Verify that an unknown block returns the missing-block placeholder."""
    fake = Block(namespace="minecraft", base_name="does_not_exist_xyz")
    model = manager.get_block_model(fake)
    missing = manager.missing_block
    assert model == missing, "Unknown block should return the missing block model"
    print(f"\n[OK] Unknown block correctly returns missing_block placeholder")
    print(f"     Missing block textures: {missing.textures}")


def main():
    print("=" * 60)
    print("Minecraft Model Reader — Sample Model Loading Test")
    print("=" * 60)

    test_load_resource_pack()
    print()

    manager = test_load_resource_pack_manager()

    test_get_block_models(manager)
    print()

    test_missing_block(manager)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
