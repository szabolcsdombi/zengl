import os


def get(name):
    return os.path.normpath(os.path.join(__file__, '..', 'assets', name))
