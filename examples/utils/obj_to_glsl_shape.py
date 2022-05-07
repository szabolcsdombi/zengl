import numpy as np

lines = open('untitled.obj', 'r').read().splitlines()

vertices = [x.replace('v ', 'vec3(').replace(' ', ', ') + ')' for x in lines if x.startswith('v ')]
normals = [x.replace('vn ', 'vec3(').replace(' ', ', ') + ')' for x in lines if x.startswith('vn ')]
texcoords = [x.replace('vt ', 'vec2(').replace(' ', ', ') + ')' for x in lines if x.startswith('vt ')]
faces = np.array([list(map(int, x.replace('f ', '').replace('/', ' ').split())) for x in lines if x.startswith('f ')])

print(',\n'.join(vertices))
print(',\n'.join(normals))
print(',\n'.join(texcoords))
print(', '.join(str(x) for x in (faces[:, 0::3] - 1).flatten()))
print(', '.join(str(x) for x in (faces[:, 1::3] - 1).flatten()))
print(', '.join(str(x) for x in (faces[:, 2::3] - 1).flatten()))
