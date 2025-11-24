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
    "export_gltf_mesh",
    "export_all_formats",
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


def export_gltf_mesh(
    path: str | Path,
    terrain,
    scale: float = 10.0,
    height_scale: float = 2.0,
    embed_textures: bool = True,
) -> Path:
    """
    Export terrain as glTF 2.0 format with heightmap, normal map, and optional textures.
    
    Args:
        path: Output .gltf or .glb file path
        terrain: TerrainMaps or compatible terrain data
        scale: Horizontal scale multiplier
        height_scale: Vertical exaggeration
        embed_textures: If True, embed textures as base64 data URIs (textured glTF)
    
    Returns:
        Path to saved glTF file
    """
    import base64
    import json
    import struct
    from io import BytesIO

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    maps = _maps(terrain)
    height = maps.height
    h, w = height.shape

    # Generate vertices, normals, and UVs
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Positions
    positions = np.zeros((h * w, 3), dtype=np.float32)
    positions[:, 0] = (x_coords / w * scale).ravel()
    positions[:, 1] = (height * height_scale).ravel()
    positions[:, 2] = (y_coords / h * scale).ravel()

    # Normals
    if maps.normals is not None:
        # Use pre-computed normals from terrain
        normals_rgb = maps.normals.astype(np.float32) / 255.0 * 2.0 - 1.0
        normals = normals_rgb.reshape(-1, 3)
    else:
        # Compute simple normals from heightmap
        normals = np.zeros((h * w, 3), dtype=np.float32)
        normals[:, 1] = 1.0

    # UVs
    uv_y, uv_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    uvs = np.zeros((h * w, 2), dtype=np.float32)
    uvs[:, 0] = (uv_x / max(1, w - 1)).ravel()
    uvs[:, 1] = (uv_y / max(1, h - 1)).ravel()

    # Generate indices (two triangles per quad)
    indices = []
    for y in range(h - 1):
        for x in range(w - 1):
            v1 = y * w + x
            v2 = y * w + x + 1
            v3 = (y + 1) * w + x + 1
            v4 = (y + 1) * w + x
            indices.extend([v1, v2, v3, v1, v3, v4])

    indices = np.array(indices, dtype=np.uint32)

    # Binary buffer data
    buffer_data = BytesIO()

    # Write positions
    pos_offset = buffer_data.tell()
    positions.tofile(buffer_data)
    pos_length = buffer_data.tell() - pos_offset

    # Write normals
    norm_offset = buffer_data.tell()
    normals.tofile(buffer_data)
    norm_length = buffer_data.tell() - norm_offset

    # Write UVs
    uv_offset = buffer_data.tell()
    uvs.tofile(buffer_data)
    uv_length = buffer_data.tell() - uv_offset

    # Write indices
    idx_offset = buffer_data.tell()
    indices.tofile(buffer_data)
    idx_length = buffer_data.tell() - idx_offset

    buffer_bytes = buffer_data.getvalue()
    buffer_b64 = base64.b64encode(buffer_bytes).decode('ascii')

    # Prepare texture data URIs if embedding
    textures = []
    images = []
    samplers = [{"magFilter": 9729, "minFilter": 9987}]

    if embed_textures:
        # Heightmap texture
        height_img = Image.fromarray(maps.height_u16(), mode="I;16")
        height_io = BytesIO()
        height_img.save(height_io, format="PNG")
        height_b64 = base64.b64encode(height_io.getvalue()).decode('ascii')
        images.append({
            "uri": f"data:image/png;base64,{height_b64}",
            "mimeType": "image/png"
        })
        textures.append({"sampler": 0, "source": 0})

        # Normal map texture
        if maps.normals is not None:
            normal_img = Image.fromarray(maps.normal_map_u8(), mode="RGB")
            normal_io = BytesIO()
            normal_img.save(normal_io, format="PNG")
            normal_b64 = base64.b64encode(normal_io.getvalue()).decode('ascii')
            images.append({
                "uri": f"data:image/png;base64,{normal_b64}",
                "mimeType": "image/png"
            })
            textures.append({"sampler": 0, "source": 1})

    # Build glTF JSON structure
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "GPU Terrain Generator"
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2
                },
                "indices": 3,
                "material": 0
            }]
        }],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.75, 0.65, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9
            },
            "normalTexture": {"index": 1} if embed_textures and len(textures) > 1 else None
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist()
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(normals),
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": len(uvs),
                "type": "VEC2"
            },
            {
                "bufferView": 3,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset,
                "byteLength": pos_length, "target": 34962},
            {"buffer": 0, "byteOffset": norm_offset,
                "byteLength": norm_length, "target": 34962},
            {"buffer": 0, "byteOffset": uv_offset,
                "byteLength": uv_length, "target": 34962},
            {"buffer": 0, "byteOffset": idx_offset,
                "byteLength": idx_length, "target": 34963}
        ],
        "buffers": [{
            "byteLength": len(buffer_bytes),
            "uri": f"data:application/octet-stream;base64,{buffer_b64}"
        }]
    }

    # Add textures if embedded
    if embed_textures and textures:
        gltf["textures"] = textures
        gltf["images"] = images
        gltf["samplers"] = samplers

    # Remove None values
    if gltf["materials"][0]["normalTexture"] is None:
        del gltf["materials"][0]["normalTexture"]

    # Write glTF file
    with target.open("w", encoding="utf-8") as f:
        json.dump(gltf, f, indent=2)

    return target


def export_all_formats(
    output_dir: str | Path,
    terrain,
    base_name: str = "terrain",
    formats: list[str] | None = None,
    **kwargs
) -> dict[str, Path]:
    """
    Export terrain to multiple formats at once.
    
    Args:
        output_dir: Directory to save all exports
        terrain: TerrainMaps or compatible terrain data
        base_name: Base filename for all exports
        formats: List of formats to export. Options: 'png', 'raw', 'r32', 'obj', 'stl', 'gltf', 'npz'
                 If None, exports all formats.
        **kwargs: Additional arguments passed to export functions (scale, height_scale, etc.)
    
    Returns:
        Dictionary mapping format names to output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = ['png', 'raw', 'r32', 'obj', 'stl', 'gltf', 'npz']

    results = {}

    scale = kwargs.get('scale', 10.0)
    height_scale = kwargs.get('height_scale', 2.0)

    if 'png' in formats:
        results['heightmap_png'] = save_heightmap_png(
            output_dir / f"{base_name}_height.png", terrain
        )
        results['normal_png'] = save_normal_map_png(
            output_dir / f"{base_name}_normal.png", terrain
        )
        if _maps(terrain).erosion_mask is not None:
            results['erosion_png'] = save_erosion_mask_png(
                output_dir / f"{base_name}_erosion.png", terrain
            )

    if 'raw' in formats:
        results['heightmap_raw'] = save_heightmap_raw(
            output_dir / f"{base_name}.raw", terrain
        )

    if 'r32' in formats:
        results['heightmap_r32'] = save_heightmap_r32(
            output_dir / f"{base_name}.r32", terrain
        )

    if 'obj' in formats:
        results['mesh_obj'] = export_obj_mesh(
            output_dir / f"{base_name}.obj", terrain, scale, height_scale
        )

    if 'stl' in formats:
        results['mesh_stl'] = export_stl_mesh(
            output_dir / f"{base_name}.stl", terrain, scale, height_scale
        )

    if 'gltf' in formats:
        results['mesh_gltf'] = export_gltf_mesh(
            output_dir / f"{base_name}.gltf", terrain, scale, height_scale,
            embed_textures=kwargs.get('embed_textures', True)
        )

    if 'npz' in formats:
        results['bundle_npz'] = save_npz_bundle(
            output_dir / f"{base_name}.npz", terrain
        )

    return results
