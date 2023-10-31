import gzip
import struct

import bpy

groups = [x.name for x in bpy.context.object.vertex_groups]
rig = bpy.context.object.parent

buf = bytearray()
buf.extend(struct.pack("ii", len(groups), 60))

for i in range(60):
    bpy.context.scene.frame_set(i)
    for name in groups:
        base = rig.pose.bones[name].bone.matrix_local.inverted()
        mat = rig.pose.bones[name].matrix @ base
        x, y, z = mat.to_translation()
        rw, rx, ry, rz = mat.to_quaternion()
        buf.extend(struct.pack("3f4x4f", x, y, z, rx, ry, rz, rw))

open("output.rig.gz", "wb").write(gzip.compress(buf))
