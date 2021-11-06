import os


def glsl(name):
    with open(os.path.join(__file__, '../glsl', name)) as f:
        return f.read()
