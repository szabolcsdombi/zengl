import gzip
import struct

import bpy

mesh = bpy.context.object.data
mesh.calc_loop_triangles()
mesh.calc_normals_split()
mesh.calc_tangents()

buf = bytearray()
for triangle_loop in mesh.loop_triangles:
    for loop_index in triangle_loop.loops:
        loop = mesh.loops[loop_index]
        x, y, z = mesh.vertices[loop.vertex_index].co
        u, v = mesh.uv_layers.active.data[loop_index].uv
        nx, ny, nz = loop.normal
        tx, ty, tz = loop.tangent
        bx, by, bz = loop.bitangent
        buf.extend(struct.pack('3f3f3f3f2f', x, y, z, tx, ty, tz, bx, by, bz, nx, ny, nz, u, v))

open('output.mesh.gz', 'wb').write(gzip.compress(buf))
