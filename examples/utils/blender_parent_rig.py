import gzip
import struct

import bpy

groups = [x.name for x in bpy.context.object.vertex_groups]
rig = bpy.context.object.parent

buf = bytearray()
buf.extend(struct.pack("i", len(groups)))

for name in groups:
    mat = rig.pose.bones[name].bone.matrix_local
    x, y, z = mat.to_translation()
    rw, rx, ry, rz = mat.to_quaternion()
    buf.extend(struct.pack("3f4x4f", x, y, z, rx, ry, rz, rw))

for name in groups:
    mat = rig.pose.bones[name].matrix
    x, y, z = mat.to_translation()
    rw, rx, ry, rz = mat.to_quaternion()
    buf.extend(struct.pack("3f4x4f", x, y, z, rx, ry, rz, rw))

open("output.rig.gz", "wb").write(gzip.compress(buf))
