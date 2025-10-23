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


def generate_plane_glb(image_path: str, output_glb_path: str) -> None:
    """
    Generate a simple GLB file with a single textured quad using the provided image as the texture.
    The plane is centered at origin, size 1x1 on X-Z plane (Y up).
    """
    # Read image bytes
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Simple plane geometry (4 vertices)
    positions = [
        -0.5, 0.0, 0.5,
         0.5, 0.0, 0.5,
        -0.5, 0.0, -0.5,
         0.5, 0.0, -0.5,
    ]
    normals = [0.0, 1.0, 0.0] * 4
    uvs = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    indices = [0, 1, 2, 1, 3, 2]

    pos_bytes = _pack_floats(positions)
    normal_bytes = _pack_floats(normals)
    uv_bytes = _pack_floats(uvs)
    idx_bytes = _pack_uint16(indices)

    # Build binary blob: positions | normals | uvs | indices | image
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

    # BufferViews
    gltf.bufferViews = []
    # positions
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=pos_offset, byteLength=len(pos_bytes)))
    pos_bv = 0
    # normals
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=normal_offset, byteLength=len(normal_bytes)))
    normal_bv = 1
    # uvs
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=uv_offset, byteLength=len(uv_bytes)))
    uv_bv = 2
    # indices
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=idx_offset, byteLength=len(idx_bytes_padded)))
    idx_bv = 3
    # image
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=image_offset, byteLength=len(image_bytes_padded)))
    image_bv = 4

    # Accessors
    gltf.accessors = []
    # positions accessor
    min_pos, max_pos = _min_max_positions(positions)
    gltf.accessors.append(Accessor(bufferView=pos_bv, byteOffset=0, componentType=5126, count=4, type="VEC3", min=min_pos, max=max_pos))
    pos_acc = 0
    # normals accessor
    gltf.accessors.append(Accessor(bufferView=normal_bv, byteOffset=0, componentType=5126, count=4, type="VEC3"))
    normal_acc = 1
    # uvs accessor
    gltf.accessors.append(Accessor(bufferView=uv_bv, byteOffset=0, componentType=5126, count=4, type="VEC2"))
    uv_acc = 2
    # indices accessor (unsigned short)
    gltf.accessors.append(Accessor(bufferView=idx_bv, byteOffset=0, componentType=5123, count=6, type="SCALAR"))
    idx_acc = 3

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

    # Texture
    gltf.textures = [Texture(source=img_idx)]
    tex_idx = 0

    # Material
    # Some pygltflib versions don't export PBRMetallicRoughness symbol directly.
    # Use a plain dict or set attribute on Material to avoid import-time errors.
    mat = Material(name="mat0")
    # set pbrMetallicRoughness as a dict with baseColorTexture reference
    mat.pbrMetallicRoughness = {"baseColorTexture": {"index": tex_idx}}
    gltf.materials = [mat]
    mat_idx = 0

    # Mesh + Primitive
    prim = Primitive(attributes={"POSITION": pos_acc, "NORMAL": normal_acc, "TEXCOORD_0": uv_acc}, indices=idx_acc, material=mat_idx)
    mesh = Mesh(primitives=[prim], name="plane")
    gltf.meshes = [mesh]
    mesh_idx = 0

    # Node + Scene
    node = Node(mesh=mesh_idx, name="node0")
    gltf.nodes = [node]
    scene = Scene(nodes=[0])
    gltf.scenes = [scene]
    gltf.scene = 0

    # Attach binary blob and write GLB
    gltf.set_binary_blob(binary_blob)
    gltf.save_binary(output_glb_path)

    # Ensure file exists
    if not os.path.exists(output_glb_path):
        raise RuntimeError("Falha ao gerar GLB")

    return
