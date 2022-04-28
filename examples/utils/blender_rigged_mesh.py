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
        nx, ny, nz = loop.normal
        groups = sorted(mesh.vertices[loop.vertex_index].groups, key=lambda x: -x.weight)[:4]
        total_weight = sum((x.weight for x in groups), 0.0) or 1.0
        bones = ([x.group for x in groups] + [0, 0, 0, 0])[:4]
        weights = ([x.weight / total_weight for x in groups] + [0.0, 0.0, 0.0, 0.0])[:4]
        buf.extend(struct.pack('3f3f4i4f', x, y, z, nx, ny, nz, *bones, *weights))

open('output.mesh.gz', 'wb').write(gzip.compress(buf))
