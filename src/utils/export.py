"""I/O helpers for persisting generated terrain data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import TerrainMaps

__all__ = [
    "save_heightmap_png",
    "save_normal_map_png",
    "save_erosion_mask_png",
    "save_heightmap_raw",
    "save_heightmap_r32",
    "save_npz_bundle",
    "export_obj_mesh",
    "export_stl_mesh",
]


def _maps(terrain) -> TerrainMaps:
    return TerrainMaps.ensure(terrain)


def save_heightmap_png(path: str | Path, terrain) -> Path:
    """Write a 16-bit PNG heightmap to disk."""
    target = Path(path)
    img = Image.fromarray(_maps(terrain).height_u16(), mode="I;16")
    img.save(target)
    return target


def save_normal_map_png(path: str | Path, terrain) -> Path:
    target = Path(path)
    img = Image.fromarray(_maps(terrain).normal_map_u8(), mode="RGB")
    img.save(target)
    return target


def save_erosion_mask_png(path: str | Path, terrain) -> Path:
    target = Path(path)
    img = Image.fromarray(_maps(terrain).erosion_mask_u8(), mode="L")
    img.save(target)
    return target


def save_heightmap_raw(path: str | Path, terrain) -> Path:
    """Write raw 16-bit little-endian heightmap data (Mac/Windows/Linux compatible)."""
    target = Path(path)
    data = _maps(terrain).height_u16().astype('<u2')

    with target.open("wb") as f:
        f.write(data.tobytes())
    return target


def save_heightmap_r32(path: str | Path, terrain) -> Path:
    """Write raw 32-bit float little-endian heightmap data."""
    target = Path(path)
    data = _maps(terrain).height.astype('<f4')

    with target.open("wb") as f:
        f.write(data.tobytes())
    return target


def save_npz_bundle(path: str | Path, terrain) -> Path:
    target = Path(path)
    maps = _maps(terrain)
    np.savez_compressed(
        target,
        height=maps.height,
        normals=maps.normals,
        erosion_mask=maps.erosion_channel(),
    )
    return target


def export_obj_mesh(
    path: str | Path,
    terrain,
    scale: float = 10.0,
    height_scale: float = 2.0,
) -> Path:
    """Export the terrain heightmap as a simple grid mesh OBJ."""
    target = Path(path)
    height = _maps(terrain).height
    h, w = height.shape

    # Vectorized vertex generation
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    wx = (x_coords / w * scale).ravel()
    wy = (height * height_scale).ravel()
    wz = (y_coords / h * scale).ravel()
    vertices = np.stack([wx, wy, wz], axis=1)

    # Vectorized UV generation
    uv_y, uv_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    u = (uv_x / max(1, w - 1)).ravel()
    v = (uv_y / max(1, h - 1)).ravel()
    uvs = np.stack([u, v], axis=1)

    # Generate face indices (two triangles per quad)
    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            v1 = y * w + x + 1
            v2 = y * w + x + 2
            v3 = (y + 1) * w + x + 2
            v4 = (y + 1) * w + x + 1
            faces.append((v1, v1, v2, v2, v3, v3))
            faces.append((v1, v1, v3, v3, v4, v4))

    with target.open("w", encoding="utf-8") as fh:
        fh.write("# Generated terrain mesh\n")
        fh.write(f"# Resolution: {w}x{h}\n")
        fh.write(f"# Vertices: {len(vertices)}\n")
        fh.write(f"# Faces: {len(faces)}\n\n")

        # Write vertices in bulk
        for vx, vy, vz in vertices:
            fh.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")

        # Write UVs in bulk
        for tu, tv in uvs:
            fh.write(f"vt {tu:.6f} {tv:.6f}\n")

        # Write faces
        fh.write("\n")
        for face in faces:
            fh.write(
                f"f {face[0]}/{face[1]} {face[2]}/{face[3]} {face[4]}/{face[5]}\n")

    return target


def export_stl_mesh(
    path: str | Path,
    terrain,
    scale: float = 10.0,
    height_scale: float = 2.0,
) -> Path:
    """Export the terrain heightmap as a binary STL mesh."""
    import struct
    target = Path(path)
    height = _maps(terrain).height
    h, w = height.shape

    # Vectorized vertex generation
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    wx = (x_coords / w * scale).astype(np.float32)
    wy = (height * height_scale).astype(np.float32)
    wz = (y_coords / h * scale).astype(np.float32)

    # Create quad vertices slices
    # v1=(x,y), v2=(x+1,y), v3=(x+1,y+1), v4=(x,y+1)
    v1_x, v1_y, v1_z = wx[:-1, :-1], wy[:-1, :-1], wz[:-1, :-1]
    v2_x, v2_y, v2_z = wx[:-1, 1:],  wy[:-1, 1:],  wz[:-1, 1:]
    v3_x, v3_y, v3_z = wx[1:, 1:],   wy[1:, 1:],   wz[1:, 1:]
    v4_x, v4_y, v4_z = wx[1:, :-1],  wy[1:, :-1],  wz[1:, :-1]

    # Stack for T1: v1, v2, v3
    t1_x = np.stack([v1_x, v2_x, v3_x], axis=-1)
    t1_y = np.stack([v1_y, v2_y, v3_y], axis=-1)
    t1_z = np.stack([v1_z, v2_z, v3_z], axis=-1)

    # Stack for T2: v1, v3, v4
    t2_x = np.stack([v1_x, v3_x, v4_x], axis=-1)
    t2_y = np.stack([v1_y, v3_y, v4_y], axis=-1)
    t2_z = np.stack([v1_z, v3_z, v4_z], axis=-1)

    # Flatten and concatenate
    n_quads = (h - 1) * (w - 1)

    # Reshape to (N, 3)
    t1_x = t1_x.reshape(n_quads, 3)
    t1_y = t1_y.reshape(n_quads, 3)
    t1_z = t1_z.reshape(n_quads, 3)

    t2_x = t2_x.reshape(n_quads, 3)
    t2_y = t2_y.reshape(n_quads, 3)
    t2_z = t2_z.reshape(n_quads, 3)

    # Concatenate triangles
    all_x = np.concatenate([t1_x, t2_x], axis=0)
    all_y = np.concatenate([t1_y, t2_y], axis=0)
    all_z = np.concatenate([t1_z, t2_z], axis=0)

    # Stack into (N, 3, 3) -> (Triangle, Vertex, Coord)
    triangles = np.stack([all_x, all_y, all_z], axis=2)

    # Compute normals
    v1 = triangles[:, 0, :]
    v2 = triangles[:, 1, :]
    v3 = triangles[:, 2, :]

    edge1 = v2 - v1
    edge2 = v3 - v1
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(
        normals), where=norms != 0)

    # Prepare binary STL data
    num_triangles = len(triangles)
    dtype = np.dtype([
        ('normal', '<f4', (3,)),
        ('v1', '<f4', (3,)),
        ('v2', '<f4', (3,)),
        ('v3', '<f4', (3,)),
        ('attr', '<u2'),
    ])

    data = np.zeros(num_triangles, dtype=dtype)
    data['normal'] = normals
    data['v1'] = triangles[:, 0, :]
    data['v2'] = triangles[:, 1, :]
    data['v3'] = triangles[:, 2, :]

    with target.open("wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", num_triangles))
        data.tofile(fh)

    return target
