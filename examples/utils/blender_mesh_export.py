import struct

import bpy


def color_byte(value):
    x = int(round(value * 255))
    return min(max(x, 0), 255)


def save_to_file(name, content):
    with open(bpy.path.abspath(name), "wb") as f:
        f.write(content)


def export_simple_mesh(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    mesh.calc_normals_split()

    Vertex = struct.Struct("=3f3f2f")

    buf = bytearray()
    for triangle_loop in mesh.loop_triangles:
        for loop_index in triangle_loop.loops:
            loop = mesh.loops[loop_index]
            x, y, z = mesh.vertices[loop.vertex_index].co
            u, v = mesh.uv_layers.active.data[loop_index].uv
            nx, ny, nz = loop.normal
            buf.extend(Vertex.pack(x, y, z, nx, ny, nz, u, v))

    return bytes(buf)


def export_color_mesh(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    mesh.calc_normals_split()

    Vertex = struct.Struct("=3f3f4B")

    buf = bytearray()
    for triangle_loop in mesh.loop_triangles:
        for loop_index in triangle_loop.loops:
            loop = mesh.loops[loop_index]
            x, y, z = mesh.vertices[loop.vertex_index].co
            r, g, b, _ = mesh.vertex_colors.active.data[loop_index].color
            r, g, b = color_byte(r), color_byte(g), color_byte(b)
            nx, ny, nz = loop.normal
            buf.extend(Vertex.pack(x, y, z, nx, ny, nz, r, g, b, 255))

    return bytes(buf)


def export_btn_mesh(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    mesh.calc_normals_split()
    mesh.calc_tangents()

    Vertex = struct.Struct("=3f3f3f3f2f")

    buf = bytearray()
    for triangle_loop in mesh.loop_triangles:
        for loop_index in triangle_loop.loops:
            loop = mesh.loops[loop_index]
            x, y, z = mesh.vertices[loop.vertex_index].co
            u, v = mesh.uv_layers.active.data[loop_index].uv
            nx, ny, nz = loop.normal
            tx, ty, tz = loop.tangent
            bx, by, bz = loop.bitangent
            buf.extend(Vertex.pack(x, y, z, tx, ty, tz, bx, by, bz, nx, ny, nz, u, v))

    return bytes(buf)


def export_material_node_color_mesh(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    mesh.calc_normals_split()

    Vertex = struct.Struct("=3f3f4B")

    buf = bytearray()
    for triangle_loop in mesh.loop_triangles:
        for loop_index in triangle_loop.loops:
            loop = mesh.loops[loop_index]
            shader = mesh.materials[triangle_loop.material_index].node_tree.nodes['Principled BSDF']
            x, y, z = mesh.vertices[loop.vertex_index].co
            r, g, b, _ = shader.inputs[0].default_value
            r, g, b = color_byte(r), color_byte(g), color_byte(b)
            nx, ny, nz = loop.normal
            buf.extend(Vertex.pack(x, y, z, nx, ny, nz, r, g, b, 255))

    return bytes(buf)


def export_material_color_mesh(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    mesh.calc_normals_split()

    Vertex = struct.Struct("=3f3f4B")

    buf = bytearray()
    for triangle_loop in mesh.loop_triangles:
        for loop_index in triangle_loop.loops:
            loop = mesh.loops[loop_index]
            x, y, z = mesh.vertices[loop.vertex_index].co
            r, g, b, _ = mesh.materials[triangle_loop.material_index].diffuse_color
            r, g, b = color_byte(r), color_byte(g), color_byte(b)
            nx, ny, nz = loop.normal
            buf.extend(Vertex.pack(x, y, z, nx, ny, nz, r, g, b, 255))

    return bytes(buf)


def export_material_mesh(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    mesh.calc_normals_split()

    Vertex = struct.Struct("=3f3f")

    buffers = [bytearray() for _ in obj.materials]
    for triangle_loop in mesh.loop_triangles:
        for loop_index in triangle_loop.loops:
            loop = mesh.loops[loop_index]
            x, y, z = mesh.vertices[loop.vertex_index].co
            nx, ny, nz = loop.normal
            buffers[triangle_loop.material_index].extend(Vertex.pack(x, y, z, nx, ny, nz))

    return [bytes(buf) for buf in buffers]


def export_baked_transform_mesh(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    mesh.calc_normals_split()

    Vertex = struct.Struct("=3f3f2f")

    matrix_world = obj.matrix_world
    matrix_normals = obj.matrix_world.inverted().transposed()

    buf = bytearray()
    for triangle_loop in mesh.loop_triangles:
        for loop_index in triangle_loop.loops:
            loop = mesh.loops[loop_index]
            x, y, z = matrix_world @ mesh.vertices[loop.vertex_index].co
            u, v = mesh.uv_layers.active.data[loop_index].uv
            nx, ny, nz = matrix_normals @ loop.normal
            buf.extend(Vertex.pack(x, y, z, nx, ny, nz, u, v))

    return bytes(buf)


def export_world_transform(obj):
    x, y, z = obj.matrix_world.to_translation()
    rw, rx, ry, rz = obj.matrix_world.to_quaternion()
    sx, sy, sz = obj.matrix_world.to_scale()
    return {
        'position': (x, y, z),
        'rotation': (rx, ry, rz, rw),
        'scale': (sx, sy, sz),
    }


def export_collection(name, exporter, filter=None):
    objects = bpy.data.collections[name].objects
    if filter:
        objects = [obj for obj in objects if filter(obj)]
    return [exporter(obj) for obj in objects]
