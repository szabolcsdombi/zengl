import os


def glsl(name):
    with open(os.path.normpath(os.path.join(os.path.abspath(__file__), '../glsl', name))) as f:
        return f.read()
