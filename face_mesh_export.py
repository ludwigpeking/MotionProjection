"""Export the tracked face as a UV-unwrapped OBJ for painting in Blender.

Topology and UV layout come from MediaPipe's canonical face model
(canonical_face_model.obj: 468 vertices, 898 triangles, one UV per vertex),
so the exported mesh uses the same unwrap as every other MediaPipe-based
face texture. Vertex positions are the live 3D landmarks; the texture is
the camera frame baked into that UV layout.

A texture painted in Blender on face_texture.png (same file name and layout)
can then be projected back through the live mesh with main.py --paint.
"""

import os

import cv2
import numpy

from face_mesh_render import render_face_mesh

CANONICAL_MODEL_PATH = "canonical_face_model.obj"
IDENTITY_UV_PROJECTION = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


_TOPOLOGY_CACHE = {}


def load_canonical_topology(path=CANONICAL_MODEL_PATH):
    """Return (uv_per_vertex (468, 2) with v up, triangles (898, 3) 0-based). Parsed once per path."""
    if path in _TOPOLOGY_CACHE:
        return _TOPOLOGY_CACHE[path]
    texture_coordinates = []
    faces = []
    with open(path) as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "vt":
                texture_coordinates.append((float(parts[1]), float(parts[2])))
            elif parts[0] == "f":
                faces.append([(int(token.split("/")[0]) - 1, int(token.split("/")[1]) - 1) for token in parts[1:4]])
    texture_coordinates = numpy.array(texture_coordinates, dtype=numpy.float64)
    uv_per_vertex = numpy.full((468, 2), numpy.nan)
    for face in faces:
        for vertex_index, uv_index in face:
            uv_per_vertex[vertex_index] = texture_coordinates[uv_index]
    triangles = numpy.array([[vertex_index for vertex_index, _ in face] for face in faces], dtype=numpy.int32)
    _TOPOLOGY_CACHE[path] = (uv_per_vertex, triangles)
    return uv_per_vertex, triangles


def uv_to_texture_pixels(uv_per_vertex, texture_size):
    """OBJ v runs bottom-up; image rows run top-down."""
    return numpy.column_stack([uv_per_vertex[:, 0] * (texture_size - 1),
                               (1.0 - uv_per_vertex[:, 1]) * (texture_size - 1)])


def bake_texture(frame_bgr, landmarks_2d, uv_per_vertex, triangles, texture_size):
    """Camera frame -> UV texture: every triangle's camera pixels warped into its UV triangle."""
    uv_pixels = uv_to_texture_pixels(uv_per_vertex, texture_size)
    uv_as_3d = numpy.column_stack([uv_pixels, numpy.zeros(len(uv_pixels))])
    return render_face_mesh(frame_bgr, landmarks_2d, uv_as_3d, IDENTITY_UV_PROJECTION, triangles,
                            texture_size, texture_size)


def landmarks_to_blender(landmarks_3d, scale=0.01):
    """Camera pixel units -> Blender: x right, y up, z toward the viewer (MediaPipe z grows away)."""
    centred = landmarks_3d - landmarks_3d.mean(axis=0)
    return numpy.column_stack([centred[:, 0], -centred[:, 1], -centred[:, 2]]) * scale


def export_face_mesh(output_directory, landmarks_3d, frame_bgr, landmarks_2d, projection=None, texture_size=1024):
    """Write face_mesh.obj, face_mesh.mtl, face_texture.png and face_mesh_state.npz. Returns the OBJ path."""
    os.makedirs(output_directory, exist_ok=True)
    uv_per_vertex, triangles = load_canonical_topology()
    vertices = landmarks_to_blender(landmarks_3d)

    obj_path = os.path.join(output_directory, "face_mesh.obj")
    with open(obj_path, "w") as file:
        file.write("# MediaPipe face mesh, live landmarks, canonical UV layout\n")
        file.write("mtllib face_mesh.mtl\no face\n")
        for x, y, z in vertices:
            file.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
        for u, v in uv_per_vertex:
            file.write(f"vt {u:.6f} {v:.6f}\n")
        file.write("usemtl face\ns off\n")
        for a, b, c in triangles:
            file.write(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n")
    with open(os.path.join(output_directory, "face_mesh.mtl"), "w") as file:
        file.write("newmtl face\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd face_texture.png\n")

    texture = bake_texture(frame_bgr, landmarks_2d, uv_per_vertex, triangles, texture_size)
    cv2.imwrite(os.path.join(output_directory, "face_texture.png"), texture)
    numpy.savez(os.path.join(output_directory, "face_mesh_state.npz"),
                landmarks_3d=landmarks_3d, landmarks_2d=landmarks_2d,
                projection=projection if projection is not None else numpy.zeros((3, 4)))
    return obj_path


def load_painted_texture(path):
    texture = cv2.imread(path, cv2.IMREAD_COLOR)
    if texture is None:
        raise FileNotFoundError(path)
    return texture


def render_painted_texture(painted_texture, landmarks_3d, projection, width, height, viewpoint=None):
    """Project a texture painted in the canonical UV layout through the live mesh (needs P).
    With a viewpoint (the projector's position) only the triangles facing it are drawn."""
    uv_per_vertex, triangles = load_canonical_topology()
    uv_pixels = uv_to_texture_pixels(uv_per_vertex, painted_texture.shape[0])
    return render_face_mesh(painted_texture, uv_pixels, landmarks_3d, projection, triangles, width, height,
                            viewpoint=viewpoint)
