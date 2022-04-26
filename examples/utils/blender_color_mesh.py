import gzip
import struct

import bpy

mesh = bpy.context.object.data
mesh.calc_loop_triangles()
mesh.calc_normals_split()

buf = bytearray()
for triangle_loop in mesh.loop_triangles:
    for loop_index in triangle_loop.loops:
        loop = mesh.loops[loop_index]
        x, y, z = mesh.vertices[loop.vertex_index].co
        r, g, b, _ = mesh.vertex_colors.active.data[loop_index].color
        nx, ny, nz = loop.normal
        buf.extend(struct.pack('3f3f3f', x, y, z, nx, ny, nz, r, g, b))

open('output.mesh.gz', 'wb').write(gzip.compress(buf))
