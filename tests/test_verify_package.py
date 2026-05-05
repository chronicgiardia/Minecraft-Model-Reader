"""Verify minecraft-model-reader package functionality.

Run:
    python tests/test_verify_package.py
"""

import os
import sys
import traceback

VANILLA_FIX_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "minecraft_model_reader",
    "api",
    "resource_pack",
    "java",
    "java_vanilla_fix",
)

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ---------- 1. Core imports ----------
def test_imports():
    import numpy
    import amulet_nbt
    import minecraft_model_reader
    from minecraft_model_reader.api import Block, BlockMesh
    from minecraft_model_reader.api.resource_pack import (
        load_resource_pack,
        load_resource_pack_manager,
    )
    assert minecraft_model_reader.__version__, "Version string is empty"


# ---------- 2. Dependency compatibility ----------
def test_numpy_amulet_nbt_compat():
    import numpy
    import amulet_nbt
    # amulet-nbt is compiled against numpy <2.0; verify it loads
    _ = amulet_nbt.TAG_String("test")
    assert int(numpy.__version__.split(".")[0]) < 2, (
        f"numpy {numpy.__version__} is >=2.0, incompatible with amulet-nbt"
    )


# ---------- 3. Resource pack loading ----------
def test_load_java_resource_pack():
    from minecraft_model_reader.api.resource_pack import load_resource_pack
    from minecraft_model_reader.api.resource_pack.java import JavaResourcePack

    pack = load_resource_pack(VANILLA_FIX_PATH)
    assert isinstance(pack, JavaResourcePack), f"Expected JavaResourcePack, got {type(pack)}"
    assert pack.valid_pack, "Pack should be valid"


def test_load_resource_pack_manager():
    from minecraft_model_reader.api.resource_pack import load_resource_pack_manager
    from minecraft_model_reader.api.resource_pack.java import JavaResourcePackManager

    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    assert isinstance(manager, JavaResourcePackManager)
    assert len(manager.pack_paths) > 0


# ---------- 4. Block model loading ----------
def test_block_model_stone():
    import amulet_nbt
    from minecraft_model_reader.api import Block, BlockMesh
    from minecraft_model_reader.api.resource_pack import load_resource_pack_manager

    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    block = Block(namespace="minecraft", base_name="stone")
    model = manager.get_block_model(block)

    assert isinstance(model, BlockMesh)
    assert len(model.textures) > 0, "Stone should have textures"
    assert len(model.faces) > 0, "Stone should have faces"


def test_block_model_stairs():
    import amulet_nbt
    from minecraft_model_reader.api import Block, BlockMesh
    from minecraft_model_reader.api.resource_pack import load_resource_pack_manager

    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    block = Block(
        namespace="minecraft",
        base_name="oak_stairs",
        properties={
            "facing": amulet_nbt.TAG_String("north"),
            "half": amulet_nbt.TAG_String("bottom"),
            "shape": amulet_nbt.TAG_String("straight"),
            "waterlogged": amulet_nbt.TAG_String("false"),
        },
    )
    model = manager.get_block_model(block)
    assert isinstance(model, BlockMesh)
    assert len(model.faces) > 0, "Stairs should have faces"


# ---------- 5. Mesh data validation ----------
def test_mesh_data_shapes():
    import numpy
    from minecraft_model_reader.api import Block, BlockMesh
    from minecraft_model_reader.api.resource_pack import load_resource_pack_manager

    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    block = Block(namespace="minecraft", base_name="dirt")
    model = manager.get_block_model(block)

    for direction, verts in model.verts.items():
        assert isinstance(verts, numpy.ndarray), f"Verts for {direction} not ndarray"
        assert verts.shape[0] % 3 == 0, f"Verts for {direction} not divisible by 3"

    for direction, tc in model.texture_coords.items():
        assert isinstance(tc, numpy.ndarray)
        assert tc.shape[0] % 2 == 0, f"Tex coords for {direction} not divisible by 2"

    for direction, faces in model.faces.items():
        assert isinstance(faces, numpy.ndarray)
        assert faces.shape[0] % model.face_mode == 0


# ---------- 6. Missing block fallback ----------
def test_missing_block_fallback():
    from minecraft_model_reader.api import Block, BlockMesh
    from minecraft_model_reader.api.resource_pack import load_resource_pack_manager

    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    fake = Block(namespace="minecraft", base_name="does_not_exist_xyz")
    model = manager.get_block_model(fake)
    missing = manager.missing_block
    assert model == missing, "Unknown block should return the missing block model"


# ---------- 7. BlockMesh rotate ----------
def test_block_mesh_rotate():
    from minecraft_model_reader.api import Block, BlockMesh
    from minecraft_model_reader.api.resource_pack import load_resource_pack_manager

    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    block = Block(namespace="minecraft", base_name="stone")
    model = manager.get_block_model(block)
    rotated = model.rotate(1, 0)
    assert isinstance(rotated, BlockMesh)


# ---------- Run all ----------
def main():
    print("=" * 50)
    print("minecraft-model-reader — Package Verification")
    print("=" * 50)

    tests = [
        ("Core imports", test_imports),
        ("numpy / amulet-nbt compatibility", test_numpy_amulet_nbt_compat),
        ("Load Java resource pack", test_load_java_resource_pack),
        ("Load resource pack manager", test_load_resource_pack_manager),
        ("Block model: stone", test_block_model_stone),
        ("Block model: oak_stairs", test_block_model_stairs),
        ("Mesh data shapes", test_mesh_data_shapes),
        ("Missing block fallback", test_missing_block_fallback),
        ("BlockMesh.rotate()", test_block_mesh_rotate),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
