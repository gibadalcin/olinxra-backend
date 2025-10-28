import os
import struct
import math
from typing import Tuple

from pygltflib import GLTF2, Buffer, BufferView, Accessor, Scene, Node, Mesh, Primitive, Image, Texture, Material


def _pack_floats(floats):
    return struct.pack('<' + 'f' * len(floats), *floats)


def _pack_uint16(ints):
    return struct.pack('<' + 'H' * len(ints), *ints)


def _min_max_positions(positions):
    xs = positions[0::3]
    ys = positions[1::3]
    zs = positions[2::3]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def generate_plane_glb(image_path: str, output_glb_path: str, base_y: float = 0.0, plane_height: float = 1.0, flip_u: bool = True, flip_v: bool = True) -> None:
    """
    Generate a simple GLB file with a single textured quad using the provided image as the texture.
    The plane is placed upright (standing) with its base on the ground plane.
    Coordinates: X across, Y up, Z forward. The quad covers X in [-0.5,0.5] and Y in [0,1], at Z=0.
    """
    # Read image bytes
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Simple plane geometry (4 vertices)
    # Vertices: bottom-left, bottom-right, top-left, top-right
    # base_y controls the base position on the Y axis (so base_y=2 -> plane sits 2 meters above ground)
    bottom_y = float(base_y)
    top_y = bottom_y + float(plane_height)
    positions = [
        -0.5, bottom_y, 0.0,   # bottom-left
         0.5, bottom_y, 0.0,   # bottom-right
        -0.5, top_y,    0.0,   # top-left
         0.5, top_y,    0.0,   # top-right
    ]
    # Normals pointing forward along +Z so the texture faces the viewer when placed at Z=0
    normals = [0.0, 0.0, 1.0] * 4
    # UVs base: (u,v) for bottom-left, bottom-right, top-left, top-right
    base_uvs = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    # Apply optional flips to correct orientation (some viewers/images require flipping)
    uvs = []
    for i in range(0, len(base_uvs), 2):
        u = base_uvs[i]
        v = base_uvs[i+1]
        if flip_u:
            u = 1.0 - u
        if flip_v:
            v = 1.0 - v
        uvs.extend([u, v])
    indices = [0, 1, 2, 1, 3, 2]

    pos_bytes = _pack_floats(positions)
    normal_bytes = _pack_floats(normals)
    uv_bytes = _pack_floats(uvs)
    idx_bytes = _pack_uint16(indices)

    # Prepare a small backing plane (solid color) slightly behind the textured
    # plane to act as a visual background when the viewer looks from behind.
    back_offset = -0.001  # small offset in Z to sit behind the front plane
    positions_back = [
        -0.5, bottom_y, back_offset,   # bottom-left (back)
         0.5, bottom_y, back_offset,   # bottom-right (back)
        -0.5, top_y,    back_offset,   # top-left (back)
         0.5, top_y,    back_offset,   # top-right (back)
    ]
    normals_back = [0.0, 0.0, -1.0] * 4
    # UVs for back plane not needed for a solid color; supply zeros
    uvs_back = [0.0, 0.0] * 4
    # indices for back plane must index into the back plane's own vertex accessor (0..3)
    indices_back = [0, 1, 2, 1, 3, 2]

    # Build binary blob: positions | normals | uvs | indices | positions_back | normals_back | uvs_back | indices_back | image
    pos_bytes = _pack_floats(positions)
    normal_bytes = _pack_floats(normals)
    uv_bytes = _pack_floats(uvs)
    idx_bytes = _pack_uint16(indices)

    pos_back_bytes = _pack_floats(positions_back)
    normal_back_bytes = _pack_floats(normals_back)
    uv_back_bytes = _pack_floats(uvs_back)
    idx_back_bytes = _pack_uint16(indices_back)

    offset = 0
    blob_parts = []
    blob_parts.append(pos_bytes); pos_offset = offset; offset += len(pos_bytes)
    blob_parts.append(normal_bytes); normal_offset = offset; offset += len(normal_bytes)
    blob_parts.append(uv_bytes); uv_offset = offset; offset += len(uv_bytes)
    # indices must be aligned to 4 bytes per glTF spec for bufferViews convenience; pad if needed
    if len(idx_bytes) % 4 != 0:
        padding_len = 4 - (len(idx_bytes) % 4)
        idx_bytes_padded = idx_bytes + (b'\x00' * padding_len)
    else:
        idx_bytes_padded = idx_bytes
    blob_parts.append(idx_bytes_padded); idx_offset = offset; offset += len(idx_bytes_padded)

    # back plane bytes
    blob_parts.append(pos_back_bytes); pos_back_offset = offset; offset += len(pos_back_bytes)
    blob_parts.append(normal_back_bytes); normal_back_offset = offset; offset += len(normal_back_bytes)
    blob_parts.append(uv_back_bytes); uv_back_offset = offset; offset += len(uv_back_bytes)
    if len(idx_back_bytes) % 4 != 0:
        padding_len_back = 4 - (len(idx_back_bytes) % 4)
        idx_back_bytes_padded = idx_back_bytes + (b'\x00' * padding_len_back)
    else:
        idx_back_bytes_padded = idx_back_bytes
    blob_parts.append(idx_back_bytes_padded); idx_back_offset = offset; offset += len(idx_back_bytes_padded)

    # image align to 4 bytes as well
    if len(image_bytes) % 4 != 0:
        padding_len = 4 - (len(image_bytes) % 4)
        image_bytes_padded = image_bytes + (b'\x00' * padding_len)
    else:
        image_bytes_padded = image_bytes
    blob_parts.append(image_bytes_padded); image_offset = offset; offset += len(image_bytes_padded)

    binary_blob = b''.join(blob_parts)

    gltf = GLTF2()

    # Single buffer
    gltf.buffers = [Buffer(byteLength=len(binary_blob))]

    # BufferViews (front plane, back plane, image)
    gltf.bufferViews = []
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=pos_offset, byteLength=len(pos_bytes)))
    pos_bv = 0
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=normal_offset, byteLength=len(normal_bytes)))
    normal_bv = 1
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=uv_offset, byteLength=len(uv_bytes)))
    uv_bv = 2
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=idx_offset, byteLength=len(idx_bytes_padded)))
    idx_bv = 3

    # back plane bufferViews
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=pos_back_offset, byteLength=len(pos_back_bytes)))
    pos_back_bv = 4
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=normal_back_offset, byteLength=len(normal_back_bytes)))
    normal_back_bv = 5
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=uv_back_offset, byteLength=len(uv_back_bytes)))
    uv_back_bv = 6
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=idx_back_offset, byteLength=len(idx_back_bytes_padded)))
    idx_back_bv = 7

    # image bufferView
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=image_offset, byteLength=len(image_bytes_padded)))
    image_bv = 8

    # Accessors
    gltf.accessors = []
    # front positions accessor
    min_pos, max_pos = _min_max_positions(positions)
    gltf.accessors.append(Accessor(bufferView=pos_bv, byteOffset=0, componentType=5126, count=4, type="VEC3", min=min_pos, max=max_pos))
    pos_acc = 0
    # front normals accessor
    gltf.accessors.append(Accessor(bufferView=normal_bv, byteOffset=0, componentType=5126, count=4, type="VEC3"))
    normal_acc = 1
    # front uvs accessor
    gltf.accessors.append(Accessor(bufferView=uv_bv, byteOffset=0, componentType=5126, count=4, type="VEC2"))
    uv_acc = 2
    # front indices accessor (unsigned short)
    gltf.accessors.append(Accessor(bufferView=idx_bv, byteOffset=0, componentType=5123, count=6, type="SCALAR"))
    idx_acc = 3

    # back positions accessor
    min_pos_b, max_pos_b = _min_max_positions(positions_back)
    gltf.accessors.append(Accessor(bufferView=pos_back_bv, byteOffset=0, componentType=5126, count=4, type="VEC3", min=min_pos_b, max=max_pos_b))
    pos_back_acc = 4
    # back normals accessor
    gltf.accessors.append(Accessor(bufferView=normal_back_bv, byteOffset=0, componentType=5126, count=4, type="VEC3"))
    normal_back_acc = 5
    # back uvs accessor
    gltf.accessors.append(Accessor(bufferView=uv_back_bv, byteOffset=0, componentType=5126, count=4, type="VEC2"))
    uv_back_acc = 6
    # back indices accessor
    gltf.accessors.append(Accessor(bufferView=idx_back_bv, byteOffset=0, componentType=5123, count=6, type="SCALAR"))
    idx_back_acc = 7

    # Image
    # Try to detect mime type
    mime = 'image/png'
    lower = image_path.lower()
    if lower.endswith('.jpg') or lower.endswith('.jpeg'):
        mime = 'image/jpeg'
    elif lower.endswith('.webp'):
        mime = 'image/webp'

    gltf.images = [Image(mimeType=mime, bufferView=image_bv, name="texture0")]
    img_idx = 0

    # Texture (for front material)
    gltf.textures = [Texture(source=img_idx)]
    tex_idx = 0

    # Materials
    # Front material: uses the texture
    mat = Material(name="mat0")
    mat.pbrMetallicRoughness = {"baseColorTexture": {"index": tex_idx}}
    try:
        mat.doubleSided = True
    except Exception:
        setattr(mat, 'doubleSided', True)

    # Back material: solid color (light gray) for backing plane
    mat_back = Material(name="mat_back")
    # baseColorFactor = [R,G,B,A]
    mat_back.pbrMetallicRoughness = {"baseColorFactor": [0.95, 0.95, 0.95, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}
    try:
        mat_back.doubleSided = True
    except Exception:
        setattr(mat_back, 'doubleSided', True)

    gltf.materials = [mat, mat_back]

    # Meshes + Primitives
    prim_front = Primitive(attributes={"POSITION": pos_acc, "NORMAL": normal_acc, "TEXCOORD_0": uv_acc}, indices=idx_acc, material=0)
    mesh_front = Mesh(primitives=[prim_front], name="plane_front")

    prim_back = Primitive(attributes={"POSITION": pos_back_acc, "NORMAL": normal_back_acc}, indices=idx_back_acc, material=1)
    mesh_back = Mesh(primitives=[prim_back], name="plane_back")

    gltf.meshes = [mesh_front, mesh_back]

    # Nodes + Scene: include both meshes so they render together
    node_front = Node(mesh=0, name="node_front")
    node_back = Node(mesh=1, name="node_back")
    gltf.nodes = [node_front, node_back]
    scene = Scene(nodes=[0, 1])
    gltf.scenes = [scene]
    gltf.scene = 0

    # Attach binary blob and write GLB
    gltf.set_binary_blob(binary_blob)
    gltf.save_binary(output_glb_path)

    # Ensure file exists
    if not os.path.exists(output_glb_path):
        raise RuntimeError("Falha ao gerar GLB")

    return
