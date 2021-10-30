from setuptools import Extension, setup

ext = Extension(
    name='zengl',
    sources=['zengl.cpp'],
    depends=['zengl.hpp'],
    extra_compile_args=['-fpermissive'],
)

with open('README.md') as readme:
    long_description = readme.read()

setup(
    name='zengl',
    version='0.3.0',
    py_modules=['_zengl'],
    data_files=[('.', ['zengl.pyi'])],
    ext_modules=[ext],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/szabolcsdombi/zengl/',
    author='Szabolcs Dombi',
    author_email='cprogrammer1994@gmail.com',
    license='MIT',
)
