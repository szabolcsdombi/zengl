from setuptools import Extension, setup

ext = Extension(
    name='zengl',
    sources=['zengl.cpp'],
    depends=['zengl.hpp'],
    extra_compile_args=['-fpermissive'],
)

setup(
    name='zengl',
    version='0.2.0',
    py_modules=['_zengl'],
    ext_modules=[ext],
)
