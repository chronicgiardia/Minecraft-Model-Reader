"""Test BlockMesh data structure integrity and numpy array immutability.

Run:
    python tests/test_blockmesh.py
"""

import os
import sys
import traceback

import numpy

from minecraft_model_reader.api.mesh.block.block_mesh import (
    BlockMesh,
    Transparency,
    FACE_KEYS,
    cull_remap_all,
)
from minecraft_model_reader.api.mesh.block.cube import get_cube, get_unit_cube

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stone_cube() -> BlockMesh:
    """Create a simple opaque unit cube."""
    tex = "minecraft:block/stone"
    return get_unit_cube(tex, tex, tex, tex, tex, tex, Transparency.FullOpaque)


def _make_glass_cube() -> BlockMesh:
    """Create a translucent unit cube with a different texture."""
    tex = "minecraft:block/glass"
    return get_unit_cube(tex, tex, tex, tex, tex, tex, Transparency.FullTranslucent)


# ---------------------------------------------------------------------------
# 1. Construction and basic properties
# ---------------------------------------------------------------------------

def test_unit_cube_properties():
    mesh = _make_stone_cube()
    assert mesh.face_mode == 3, "face_mode should be 3 (triangles)"
    assert mesh.is_transparent == Transparency.FullOpaque
    assert mesh.is_opaque is True
    assert len(mesh.textures) == 1
    assert mesh.textures[0] == "minecraft:block/stone"


def test_transparency_enum_ordering():
    assert Transparency.FullOpaque < Transparency.FullTranslucent < Transparency.Partial


def test_face_keys_present():
    """A unit cube should have one entry per cardinal direction (no None key)."""
    mesh = _make_stone_cube()
    for direction in ("down", "up", "north", "east", "south", "west"):
        assert direction in mesh.faces, f"Missing face direction: {direction}"
    assert None not in mesh.faces, "Unit cube should not have un-culled faces"


# ---------------------------------------------------------------------------
# 2. Array shape invariants
# ---------------------------------------------------------------------------

def test_verts_shape():
    mesh = _make_stone_cube()
    for direction, verts in mesh.verts.items():
        assert isinstance(verts, numpy.ndarray)
        assert verts.ndim == 1, f"verts[{direction}] should be 1-D"
        assert verts.shape[0] % 3 == 0, f"verts[{direction}] length not multiple of 3"


def test_texture_coords_shape():
    mesh = _make_stone_cube()
    for direction, tc in mesh.texture_coords.items():
        assert isinstance(tc, numpy.ndarray)
        assert tc.ndim == 1
        assert tc.shape[0] % 2 == 0, f"texture_coords[{direction}] length not multiple of 2"


def test_tint_verts_shape():
    mesh = _make_stone_cube()
    for direction, tv in mesh.tint_verts.items():
        assert isinstance(tv, numpy.ndarray)
        assert tv.ndim == 1
        assert tv.shape[0] % 3 == 0, f"tint_verts[{direction}] length not multiple of 3"


def test_faces_shape():
    mesh = _make_stone_cube()
    for direction, faces in mesh.faces.items():
        assert isinstance(faces, numpy.ndarray)
        assert numpy.issubdtype(faces.dtype, numpy.unsignedinteger)
        assert faces.shape[0] % mesh.face_mode == 0


def test_texture_index_matches_faces():
    mesh = _make_stone_cube()
    for direction in mesh.faces:
        n_faces = mesh.faces[direction].shape[0] // mesh.face_mode
        n_tex = mesh.texture_index[direction].shape[0]
        assert n_tex == n_faces, (
            f"texture_index[{direction}] count ({n_tex}) != face count ({n_faces})"
        )


def test_vert_count_consistency():
    """Number of vertices implied by verts, texture_coords, and tint_verts should agree."""
    mesh = _make_stone_cube()
    for direction in mesh.verts:
        n_verts = mesh.verts[direction].shape[0] // 3
        n_tc = mesh.texture_coords[direction].shape[0] // 2
        n_tv = mesh.tint_verts[direction].shape[0] // 3
        assert n_verts == n_tc == n_tv, (
            f"Vertex count mismatch for {direction}: verts={n_verts}, tc={n_tc}, tint={n_tv}"
        )


# ---------------------------------------------------------------------------
# 3. Numpy array immutability
# ---------------------------------------------------------------------------

def test_verts_immutable():
    mesh = _make_stone_cube()
    for direction, arr in mesh.verts.items():
        try:
            arr[0] = 999.0
            raise AssertionError(f"verts[{direction}] is writable — should be read-only")
        except ValueError:
            pass  # expected: assignment destination is read-only


def test_texture_coords_immutable():
    mesh = _make_stone_cube()
    for direction, arr in mesh.texture_coords.items():
        try:
            arr[0] = 999.0
            raise AssertionError(f"texture_coords[{direction}] is writable")
        except ValueError:
            pass


def test_faces_immutable():
    mesh = _make_stone_cube()
    for direction, arr in mesh.faces.items():
        try:
            arr[0] = 999
            raise AssertionError(f"faces[{direction}] is writable")
        except ValueError:
            pass


def test_texture_index_immutable():
    mesh = _make_stone_cube()
    for direction, arr in mesh.texture_index.items():
        try:
            arr[0] = 999
            raise AssertionError(f"texture_index[{direction}] is writable")
        except ValueError:
            pass


def test_vert_tables_immutable():
    """vert_tables is lazily built — verify it's also read-only once accessed."""
    mesh = _make_stone_cube()
    for direction, arr in mesh.vert_tables.items():
        try:
            arr[0] = 999.0
            raise AssertionError(f"vert_tables[{direction}] is writable")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# 4. Merge
# ---------------------------------------------------------------------------

def test_merge_single():
    """Merging a single model should produce an equivalent mesh."""
    original = _make_stone_cube()
    merged = BlockMesh.merge([original])
    assert merged == original


def test_merge_multiple():
    stone = _make_stone_cube()
    glass = _make_glass_cube()
    merged = BlockMesh.merge([stone, glass])

    # Should contain both texture paths
    assert "minecraft:block/stone" in merged.textures
    assert "minecraft:block/glass" in merged.textures

    # Transparency should be min(FullOpaque, FullTranslucent) = FullOpaque
    assert merged.is_transparent == Transparency.FullOpaque


def test_merge_empty():
    """Merging an empty iterable should still produce a valid BlockMesh."""
    merged = BlockMesh.merge([])
    assert isinstance(merged, BlockMesh)
    assert len(merged.textures) == 0
    assert len(merged.faces) == 0


def test_merge_preserves_immutability():
    merged = BlockMesh.merge([_make_stone_cube(), _make_glass_cube()])
    for direction, arr in merged.verts.items():
        try:
            arr[0] = 999.0
            raise AssertionError(f"Merged verts[{direction}] is writable")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# 5. Rotate
# ---------------------------------------------------------------------------

def test_rotate_identity():
    """rotate(0, 0) should return the same object."""
    mesh = _make_stone_cube()
    rotated = mesh.rotate(0, 0)
    assert rotated is mesh, "rotate(0,0) should return self"


def test_rotate_produces_valid_mesh():
    mesh = _make_stone_cube()
    rotated = mesh.rotate(1, 0)
    assert isinstance(rotated, BlockMesh)
    # Rotated mesh should have the same number of cull directions
    assert set(rotated.faces.keys()) == set(mesh.faces.keys())


def test_rotate_remaps_cull_directions():
    """A 90° Y rotation should cycle north→east→south→west."""
    mesh = _make_stone_cube()
    rotated = mesh.rotate(0, 1)
    # The cull remap for (roty=1, rotx=0) maps north→east, east→south, etc.
    expected_remap = cull_remap_all[(1, 0)]
    for orig_dir in mesh.faces:
        remapped = expected_remap[orig_dir]
        assert remapped in rotated.faces, (
            f"Expected remapped direction {remapped} (from {orig_dir}) in rotated mesh"
        )


def test_rotate_preserves_textures():
    mesh = _make_stone_cube()
    rotated = mesh.rotate(1, 1)
    assert rotated.textures == mesh.textures


def test_rotate_preserves_transparency():
    mesh = _make_stone_cube()
    rotated = mesh.rotate(1, 0)
    assert rotated.is_transparent == mesh.is_transparent


# ---------------------------------------------------------------------------
# 6. Equality
# ---------------------------------------------------------------------------

def test_equality_same():
    a = _make_stone_cube()
    b = _make_stone_cube()
    assert a == b


def test_equality_different_texture():
    a = _make_stone_cube()
    b = _make_glass_cube()
    assert a != b


def test_equality_not_implemented_for_other_types():
    mesh = _make_stone_cube()
    assert mesh.__eq__("not a mesh") is NotImplemented


# ---------------------------------------------------------------------------
# 7. get_cube with custom bounds and do_not_cull
# ---------------------------------------------------------------------------

def test_custom_bounds():
    tex = "minecraft:block/slab"
    mesh = get_cube(
        tex, tex, tex, tex, tex, tex,
        bounds=((0, 1), (0, 0.5), (0, 1)),
        transparency=Transparency.Partial,
    )
    assert mesh.is_transparent == Transparency.Partial
    assert mesh.is_opaque is False
    # Verts should stay within the specified bounds
    for direction, verts in mesh.verts.items():
        ys = verts.reshape(-1, 3)[:, 1]
        assert ys.max() <= 0.5 + 1e-9, f"Y exceeds upper bound for {direction}"
        assert ys.min() >= 0.0 - 1e-9, f"Y below lower bound for {direction}"


def test_do_not_cull():
    """Faces marked do_not_cull should be stored under None instead of their direction."""
    tex = "minecraft:block/torch"
    mesh = get_cube(
        tex, tex, tex, tex, tex, tex,
        do_not_cull=(True, True, False, False, False, False),
        transparency=Transparency.Partial,
    )
    # down and up are do_not_cull → should appear under None
    assert None in mesh.faces, "Expected None key for un-culled faces"
    assert "down" not in mesh.faces, "down should have been moved to None"
    assert "up" not in mesh.faces, "up should have been moved to None"
    # The remaining 4 cardinal directions should still be present
    for d in ("north", "east", "south", "west"):
        assert d in mesh.faces, f"{d} should still be a culled face"


# ---------------------------------------------------------------------------
# 8. Cull remap table completeness
# ---------------------------------------------------------------------------

def test_cull_remap_covers_all_rotations():
    for roty in range(-3, 4):
        for rotx in range(-3, 4):
            assert (roty, rotx) in cull_remap_all, f"Missing remap for ({roty}, {rotx})"
            remap = cull_remap_all[(roty, rotx)]
            assert remap[None] is None, "None should always map to None"
            for key in ("down", "up", "north", "east", "south", "west"):
                assert remap[key] in FACE_KEYS, (
                    f"remap[{key}] = {remap[key]} not in FACE_KEYS for ({roty}, {rotx})"
                )


# ---------------------------------------------------------------------------
# 9. Integration: load a real model and verify immutability
# ---------------------------------------------------------------------------

def test_loaded_model_structure():
    """Verify that a model loaded via the resource pack manager has correct structure."""
    from minecraft_model_reader.api import Block, BlockMesh
    from minecraft_model_reader.api.resource_pack import load_resource_pack_manager

    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    block = Block(namespace="minecraft", base_name="stone")
    model = manager.get_block_model(block)

    assert isinstance(model, BlockMesh)
    assert len(model.textures) > 0
    assert len(model.faces) > 0
    for direction, verts in model.verts.items():
        assert isinstance(verts, numpy.ndarray)
        assert verts.ndim == 1
        assert verts.shape[0] % 3 == 0
    for direction, faces in model.faces.items():
        assert isinstance(faces, numpy.ndarray)
        assert faces.shape[0] % model.face_mode == 0


def test_loaded_model_immutability():
    """Models returned by get_block_model must have write-protected arrays."""
    from minecraft_model_reader.api import Block, BlockMesh
    from minecraft_model_reader.api.resource_pack import load_resource_pack_manager

    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    block = Block(namespace="minecraft", base_name="stone")
    model = manager.get_block_model(block)

    for direction, arr in model.verts.items():
        try:
            arr[0] = 999.0
            raise AssertionError(f"Loaded model verts[{direction}] is writable")
        except ValueError:
            pass
    for direction, arr in model.faces.items():
        try:
            arr[0] = 999
            raise AssertionError(f"Loaded model faces[{direction}] is writable")
        except ValueError:
            pass
    for direction, arr in model.texture_coords.items():
        try:
            arr[0] = 999.0
            raise AssertionError(f"Loaded model texture_coords[{direction}] is writable")
        except ValueError:
            pass
    for direction, arr in model.texture_index.items():
        try:
            arr[0] = 999
            raise AssertionError(f"Loaded model texture_index[{direction}] is writable")
        except ValueError:
            pass


def test_deepcopy_preserves_immutability():
    """copy.deepcopy on a BlockMesh must preserve write-protection."""
    import copy
    mesh = _make_stone_cube()
    cloned = copy.deepcopy(mesh)
    assert cloned == mesh
    assert cloned is not mesh
    for direction, arr in cloned.verts.items():
        try:
            arr[0] = 999.0
            raise AssertionError(f"Deepcopy verts[{direction}] is writable")
        except ValueError:
            pass
    for direction, arr in cloned.faces.items():
        try:
            arr[0] = 999
            raise AssertionError(f"Deepcopy faces[{direction}] is writable")
        except ValueError:
            pass


def test_cached_model_not_mutated():
    """Successive get_block_model calls must return equal but independent objects."""
    from minecraft_model_reader.api import Block, BlockMesh
    from minecraft_model_reader.api.resource_pack import load_resource_pack_manager

    manager = load_resource_pack_manager([VANILLA_FIX_PATH])
    block = Block(namespace="minecraft", base_name="stone")
    model_a = manager.get_block_model(block)
    model_b = manager.get_block_model(block)
    assert model_a == model_b
    assert model_a is not model_b  # must be separate copies


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("BlockMesh — Data Structure & Immutability Tests")
    print("=" * 55)

    tests = [
        # Construction
        ("Unit cube properties", test_unit_cube_properties),
        ("Transparency enum ordering", test_transparency_enum_ordering),
        ("Face keys present", test_face_keys_present),
        # Array shapes
        ("Verts shape", test_verts_shape),
        ("Texture coords shape", test_texture_coords_shape),
        ("Tint verts shape", test_tint_verts_shape),
        ("Faces shape", test_faces_shape),
        ("Texture index matches faces", test_texture_index_matches_faces),
        ("Vert count consistency", test_vert_count_consistency),
        # Immutability
        ("Verts immutable", test_verts_immutable),
        ("Texture coords immutable", test_texture_coords_immutable),
        ("Faces immutable", test_faces_immutable),
        ("Texture index immutable", test_texture_index_immutable),
        ("Vert tables immutable", test_vert_tables_immutable),
        # Merge
        ("Merge single", test_merge_single),
        ("Merge multiple", test_merge_multiple),
        ("Merge empty", test_merge_empty),
        ("Merge preserves immutability", test_merge_preserves_immutability),
        # Rotate
        ("Rotate identity", test_rotate_identity),
        ("Rotate produces valid mesh", test_rotate_produces_valid_mesh),
        ("Rotate remaps cull directions", test_rotate_remaps_cull_directions),
        ("Rotate preserves textures", test_rotate_preserves_textures),
        ("Rotate preserves transparency", test_rotate_preserves_transparency),
        # Equality
        ("Equality same", test_equality_same),
        ("Equality different texture", test_equality_different_texture),
        ("Equality NotImplemented for other types", test_equality_not_implemented_for_other_types),
        # get_cube options
        ("Custom bounds", test_custom_bounds),
        ("do_not_cull", test_do_not_cull),
        # Cull remap table
        ("Cull remap table completeness", test_cull_remap_covers_all_rotations),
        # Integration
        ("Loaded model structure", test_loaded_model_structure),
        ("Loaded model immutability", test_loaded_model_immutability),
        ("Deepcopy preserves immutability", test_deepcopy_preserves_immutability),
        ("Cached model not mutated", test_cached_model_not_mutated),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print("=" * 55)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 55)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
